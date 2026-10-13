// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Checks of zstd-compressed XOR deltas against checkpoint files, off the GIL.
//
// A check decompresses a tensor's delta in chunks, records the delta's nonzero
// bytes for staging, and derives the target's Adler-32. A sparse check reads
// the checkpoint file only where the delta changes it and updates the base
// checksum: Adler-32 is linear in the bytes. A full check reads every byte.
// Bytes of deltas staged on the files are XORed in where they are read.
// Tensors run on a pool of threads while Python waits with the GIL released.

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <zstd.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kAdlerBase = 65521;
// The most bytes Adler-32's sums take before they must be reduced.
constexpr size_t kAdlerBlock = 5552;
constexpr size_t kChunk = size_t(1) << 20;

// Outcomes of one tensor's check.
enum Status : uint8_t {
  kSparse = 0,     // a sparse check hit the target checksum
  kFull = 1,       // a full check hit it, without a base checksum
  kFallback = 2,   // a sparse check missed, and a full check hit it
  kMissed = 3,     // a full check missed it
  kError = 4,      // the delta does not decompress to the tensor's size
};

struct Layer {
  const uint32_t* positions;
  const uint8_t* values;
  size_t count;
};

struct Tensor {
  size_t file;
  size_t offset;
  size_t size;
  const uint8_t* frame;
  size_t frame_size;
  int64_t base;
  uint32_t expected;
  std::vector<Layer> layers;
};

struct Result {
  Status status = kError;
  uint32_t checksum = 0;
  std::string error;
  std::vector<uint32_t> positions;
  std::vector<uint8_t> values;
};

struct Mapping {
  const uint8_t* data = nullptr;
  size_t size = 0;
};

// One thread's decompression context and buffers, reused across tensors.
struct Scratch {
  ZSTD_DCtx* context = ZSTD_createDCtx();
  std::vector<uint8_t> chunk = std::vector<uint8_t>(kChunk);
  std::vector<uint8_t> target = std::vector<uint8_t>(kChunk);
  ~Scratch() { ZSTD_freeDCtx(context); }
};

uint32_t adler32(uint32_t value, const uint8_t* data, size_t length) {
  uint32_t low = value & 0xFFFF;
  uint32_t high = value >> 16;
  while (length > 0) {
    const size_t block = std::min(length, kAdlerBlock);
    length -= block;
    for (size_t index = 0; index < block; ++index) {
      low += data[index];
      high += low;
    }
    data += block;
    low %= kAdlerBase;
    high %= kAdlerBase;
  }
  return (high << 16) | low;
}

// Staged deltas' bytes, walked forward along one tensor.
struct Overlay {
  const Layer* layer;
  size_t cursor = 0;

  // The XOR of this layer at `position`; positions must not decrease.
  uint8_t at(size_t position) {
    while (cursor < layer->count && layer->positions[cursor] < position) {
      ++cursor;
    }
    return cursor < layer->count && layer->positions[cursor] == position ? layer->values[cursor] : 0;
  }

  // XOR this layer into `bytes`, the tensor's bytes from `start`.
  void apply(uint8_t* bytes, size_t start, size_t length) {
    while (cursor < layer->count && layer->positions[cursor] < start) {
      ++cursor;
    }
    for (; cursor < layer->count && layer->positions[cursor] < start + length; ++cursor) {
      bytes[layer->positions[cursor] - start] ^= layer->values[cursor];
    }
  }
};

// Call `found(position, value)` for each nonzero byte of `block`, at tensor offset `start`.
template <typename Found>
void scan(const uint8_t* block, size_t length, size_t start, Found&& found) {
  size_t index = 0;
  for (; index < length && (reinterpret_cast<uintptr_t>(block + index) & 7) != 0; ++index) {
    if (block[index] != 0) {
      found(start + index, block[index]);
    }
  }
  for (; index + 8 <= length; index += 8) {
    uint64_t word;
    std::memcpy(&word, block + index, 8);
    if (word == 0) {
      continue;
    }
    for (size_t byte = 0; byte < 8; ++byte) {
      if (block[index + byte] != 0) {
        found(start + index + byte, block[index + byte]);
      }
    }
  }
  for (; index < length; ++index) {
    if (block[index] != 0) {
      found(start + index, block[index]);
    }
  }
}

// Decompress into `out` until it is full or the frames end.
bool decompress(ZSTD_DCtx* context, ZSTD_inBuffer& input, uint8_t* out, size_t capacity,
                size_t& written, bool& ended, std::string& error) {
  ZSTD_outBuffer output{out, capacity, 0};
  for (;;) {
    const size_t hint = ZSTD_decompressStream(context, &output, &input);
    if (ZSTD_isError(hint)) {
      error = ZSTD_getErrorName(hint);
      return false;
    }
    // The last frame decoded and flushed, with no input left, is the end.
    if (hint == 0 && input.pos == input.size) {
      written = output.pos;
      ended = true;
      return true;
    }
    if (output.pos == output.size) {
      written = output.pos;
      ended = false;
      return true;
    }
    if (input.pos == input.size) {
      error = "delta frame is truncated";
      return false;
    }
  }
}

// Run one check of `tensor`; return false with `error` set if the delta is malformed.
bool run(const Tensor& tensor, const uint8_t* old, bool sparse, Scratch& scratch, Result& result,
         std::string& error) {
  ZSTD_DCtx_reset(scratch.context, ZSTD_reset_session_only);
  ZSTD_inBuffer input{tensor.frame, tensor.frame_size, 0};
  std::vector<Overlay> overlays;
  overlays.reserve(tensor.layers.size());
  for (const Layer& layer : tensor.layers) {
    overlays.push_back(Overlay{&layer});
  }
  result.positions.clear();
  result.values.clear();
  // Sparse sums of byte changes, or the running checksum of the target.
  int64_t low = 0;
  int64_t high = 0;
  uint32_t checksum = 1;
  size_t done = 0;
  for (;;) {
    // Past the tensor's end, one more byte shows a delta that runs over.
    const size_t want = std::clamp(tensor.size - done, size_t(1), kChunk);
    size_t written = 0;
    bool ended = false;
    if (!decompress(scratch.context, input, scratch.chunk.data(), want, written, ended, error)) {
      return false;
    }
    if (done + written > tensor.size) {
      error = "delta byte size differs";
      return false;
    }
    const uint8_t* delta = scratch.chunk.data();
    if (sparse) {
      scan(delta, written, done, [&](size_t position, uint8_t value) {
        uint8_t before = old[position];
        for (Overlay& overlay : overlays) {
          before ^= overlay.at(position);
        }
        const int64_t difference = int64_t(uint8_t(before ^ value)) - int64_t(before);
        low += difference;
        high += int64_t((tensor.size - position) % kAdlerBase) * difference;
        result.positions.push_back(uint32_t(position));
        result.values.push_back(value);
      });
    } else if (written > 0) {
      uint8_t* target = scratch.target.data();
      std::memcpy(target, old + done, written);
      for (Overlay& overlay : overlays) {
        overlay.apply(target, done, written);
      }
      for (size_t index = 0; index < written; ++index) {
        target[index] ^= delta[index];
      }
      checksum = adler32(checksum, target, written);
      scan(delta, written, done, [&](size_t position, uint8_t value) {
        result.positions.push_back(uint32_t(position));
        result.values.push_back(value);
      });
    }
    done += written;
    if (ended || written < want) {
      break;
    }
  }
  if (done != tensor.size) {
    error = "delta byte size differs";
    return false;
  }
  if (sparse) {
    const int64_t base = tensor.base;
    int64_t sum = ((base & 0xFFFF) + low) % kAdlerBase;
    int64_t weighted = ((base >> 16) + high) % kAdlerBase;
    sum += sum < 0 ? kAdlerBase : 0;
    weighted += weighted < 0 ? kAdlerBase : 0;
    checksum = uint32_t((weighted << 16) | sum);
  }
  result.checksum = checksum;
  return true;
}

void check(const Tensor& tensor, const std::vector<Mapping>& maps, Scratch& scratch, Result& result) {
  const Mapping& map = maps[tensor.file];
  if (tensor.offset > map.size || tensor.size > map.size - tensor.offset) {
    result.status = kError;
    result.error = "tensor lies beyond its checkpoint file";
    return;
  }
  const uint8_t* old = map.data + tensor.offset;
  if (tensor.base >= 0) {
    if (!run(tensor, old, true, scratch, result, result.error)) {
      result.status = kError;
      return;
    }
    if (result.checksum == tensor.expected) {
      result.status = kSparse;
      return;
    }
  }
  if (!run(tensor, old, false, scratch, result, result.error)) {
    result.status = kError;
    return;
  }
  result.status = result.checksum != tensor.expected ? kMissed : tensor.base >= 0 ? kFallback : kFull;
  if (result.status == kMissed) {
    result.positions = {};
    result.values = {};
  }
}

// Run `work(index)` for each index of `order` on up to `threads` threads.
template <typename Work>
void parallel(size_t threads, const std::vector<size_t>& order, Work&& work) {
  std::atomic<size_t> next{0};
  auto drain = [&] {
    for (size_t claimed = next.fetch_add(1); claimed < order.size(); claimed = next.fetch_add(1)) {
      work(order[claimed]);
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(threads);
  for (size_t worker = 0; worker < threads; ++worker) {
    try {
      pool.emplace_back(drain);
    } catch (const std::system_error&) {
      // Fewer threads still drain every index.
      break;
    }
  }
  drain();
  for (std::thread& thread : pool) {
    thread.join();
  }
}

// Buffers of Python objects, released when the call returns.
class Views {
 public:
  ~Views() {
    for (Py_buffer& view : views_) {
      PyBuffer_Release(&view);
    }
  }

  // The contiguous buffer of `object` with items of `itemsize` bytes and one of `formats`.
  const void* take(PyObject* object, const char* formats, Py_ssize_t itemsize, size_t& count) {
    Py_buffer view;
    if (PyObject_GetBuffer(object, &view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0) {
      return nullptr;
    }
    views_.push_back(view);
    const char* format = view.format != nullptr ? view.format : "B";
    if (format[0] == '<' || format[0] == '=' || format[0] == '@') {
      ++format;
    }
    if (view.itemsize != itemsize || std::strlen(format) != 1 || std::strchr(formats, format[0]) == nullptr) {
      PyErr_Format(PyExc_ValueError, "buffer of format %s does not hold %zd-byte items", view.format,
                   itemsize);
      return nullptr;
    }
    count = size_t(view.len / view.itemsize);
    return view.buf != nullptr ? view.buf : "";
  }

 private:
  std::vector<Py_buffer> views_;
};

class Files {
 public:
  ~Files() {
    for (size_t index = 0; index < maps_.size(); ++index) {
      if (maps_[index].data != nullptr) {
        munmap(const_cast<uint8_t*>(maps_[index].data), maps_[index].size);
      }
      close(descriptors_[index]);
    }
  }

  bool open(const char* path) {
    const int descriptor = ::open(path, O_RDONLY | O_CLOEXEC);
    struct stat info;
    if (descriptor < 0 || fstat(descriptor, &info) != 0) {
      if (descriptor >= 0) {
        ::close(descriptor);
      }
      PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
      return false;
    }
    Mapping map;
    map.size = size_t(info.st_size);
    if (map.size > 0) {
      void* data = mmap(nullptr, map.size, PROT_READ, MAP_SHARED, descriptor, 0);
      if (data == MAP_FAILED) {
        ::close(descriptor);
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
        return false;
      }
      map.data = static_cast<const uint8_t*>(data);
    }
    maps_.push_back(map);
    descriptors_.push_back(descriptor);
    return true;
  }

  const std::vector<Mapping>& maps() const { return maps_; }

 private:
  std::vector<Mapping> maps_;
  std::vector<int> descriptors_;
};

PyObject* py_check(PyObject*, PyObject* args) {
  PyObject *paths, *files, *offsets, *sizes, *frames, *bases, *expected, *overlays;
  Py_ssize_t threads;
  if (!PyArg_ParseTuple(args, "OOOOOOOOn", &paths, &files, &offsets, &sizes, &frames, &bases,
                        &expected, &overlays, &threads)) {
    return nullptr;
  }
  Views views;
  size_t count = 0;
  const auto* file_index = static_cast<const int64_t*>(views.take(files, "ql", 8, count));
  if (file_index == nullptr) {
    return nullptr;
  }
  // Per-tensor arrays, each as long as `files`.
  auto column = [&](PyObject* object, const char* formats, Py_ssize_t itemsize) -> const void* {
    size_t length = 0;
    const void* data = views.take(object, formats, itemsize, length);
    if (data != nullptr && length != count) {
      PyErr_SetString(PyExc_ValueError, "per-tensor arguments differ in length");
      return nullptr;
    }
    return data;
  };
  const auto* offset = static_cast<const int64_t*>(column(offsets, "ql", 8));
  const auto* size = offset != nullptr ? static_cast<const int64_t*>(column(sizes, "ql", 8)) : nullptr;
  const auto* base = size != nullptr ? static_cast<const int64_t*>(column(bases, "ql", 8)) : nullptr;
  const auto* target = base != nullptr ? static_cast<const uint32_t*>(column(expected, "IL", 4)) : nullptr;
  if (target == nullptr) {
    return nullptr;
  }
  PyObject* frame_list = PySequence_Fast(frames, "frames must be a sequence");
  PyObject* overlay_list = frame_list != nullptr ? PySequence_Fast(overlays, "overlays must be a sequence") : nullptr;
  std::unique_ptr<PyObject, decltype(&Py_DecRef)> frame_owner(frame_list, Py_DecRef);
  std::unique_ptr<PyObject, decltype(&Py_DecRef)> overlay_owner(overlay_list, Py_DecRef);
  if (overlay_list == nullptr) {
    return nullptr;
  }
  if (PySequence_Fast_GET_SIZE(frame_list) != Py_ssize_t(count) ||
      PySequence_Fast_GET_SIZE(overlay_list) != Py_ssize_t(count)) {
    PyErr_SetString(PyExc_ValueError, "per-tensor arguments differ in length");
    return nullptr;
  }
  Files mapped;
  PyObject* path_list = PySequence_Fast(paths, "paths must be a sequence");
  std::unique_ptr<PyObject, decltype(&Py_DecRef)> path_owner(path_list, Py_DecRef);
  if (path_list == nullptr) {
    return nullptr;
  }
  for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(path_list); ++index) {
    const char* path = PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(path_list, index));
    if (path == nullptr || !mapped.open(path)) {
      return nullptr;
    }
  }
  std::vector<Tensor> tensors(count);
  for (size_t index = 0; index < count; ++index) {
    Tensor& tensor = tensors[index];
    if (file_index[index] < 0 || size_t(file_index[index]) >= mapped.maps().size() || offset[index] < 0 ||
        size[index] < 0 || size[index] > (int64_t(1) << 32)) {
      PyErr_SetString(PyExc_ValueError, "tensor file, offset, or size out of range");
      return nullptr;
    }
    tensor.file = size_t(file_index[index]);
    tensor.offset = size_t(offset[index]);
    tensor.size = size_t(size[index]);
    tensor.base = base[index];
    tensor.expected = target[index];
    tensor.frame = static_cast<const uint8_t*>(
        views.take(PySequence_Fast_GET_ITEM(frame_list, Py_ssize_t(index)), "Bbc", 1, tensor.frame_size));
    if (tensor.frame == nullptr) {
      return nullptr;
    }
    PyObject* layers = PySequence_Fast(PySequence_Fast_GET_ITEM(overlay_list, Py_ssize_t(index)),
                                       "each tensor's overlays must be a sequence");
    std::unique_ptr<PyObject, decltype(&Py_DecRef)> layer_owner(layers, Py_DecRef);
    if (layers == nullptr) {
      return nullptr;
    }
    for (Py_ssize_t layer_index = 0; layer_index < PySequence_Fast_GET_SIZE(layers); ++layer_index) {
      PyObject* pair = PySequence_Fast_GET_ITEM(layers, layer_index);
      if (!PyTuple_Check(pair) || PyTuple_GET_SIZE(pair) != 2) {
        PyErr_SetString(PyExc_ValueError, "an overlay is a (positions, values) pair");
        return nullptr;
      }
      Layer layer{nullptr, nullptr, 0};
      size_t values = 0;
      layer.positions = static_cast<const uint32_t*>(views.take(PyTuple_GET_ITEM(pair, 0), "IL", 4, layer.count));
      layer.values = layer.positions != nullptr
                         ? static_cast<const uint8_t*>(views.take(PyTuple_GET_ITEM(pair, 1), "Bb", 1, values))
                         : nullptr;
      if (layer.values == nullptr) {
        return nullptr;
      }
      if (values != layer.count) {
        PyErr_SetString(PyExc_ValueError, "staged offsets and values differ in length");
        return nullptr;
      }
      tensor.layers.push_back(layer);
    }
  }
  // Large tensors first, so one does not finish last.
  std::vector<size_t> order(count);
  std::iota(order.begin(), order.end(), size_t(0));
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t left, size_t right) { return tensors[left].size > tensors[right].size; });
  const size_t workers = size_t(std::max<Py_ssize_t>(1, std::min<Py_ssize_t>(threads, Py_ssize_t(count))));
  std::vector<Result> results(count);
  Py_BEGIN_ALLOW_THREADS;
  parallel(workers - 1, order, [&](size_t index) {
    thread_local Scratch scratch;
    try {
      check(tensors[index], mapped.maps(), scratch, results[index]);
    } catch (const std::exception& error) {
      results[index].status = kError;
      results[index].error = error.what();
      results[index].positions = {};
      results[index].values = {};
    }
  });
  Py_END_ALLOW_THREADS;
  // Each tensor's staged bytes, concatenated in argument order.
  std::vector<int64_t> counts(count);
  std::vector<size_t> starts(count);
  size_t total = 0;
  for (size_t index = 0; index < count; ++index) {
    starts[index] = total;
    counts[index] = int64_t(results[index].positions.size());
    total += results[index].positions.size();
  }
  PyObject* positions = PyByteArray_FromStringAndSize(nullptr, Py_ssize_t(total * sizeof(uint32_t)));
  PyObject* values = positions != nullptr ? PyByteArray_FromStringAndSize(nullptr, Py_ssize_t(total)) : nullptr;
  if (values == nullptr) {
    Py_XDECREF(positions);
    return nullptr;
  }
  auto* position_bytes = reinterpret_cast<uint8_t*>(PyByteArray_AS_STRING(positions));
  auto* value_bytes = reinterpret_cast<uint8_t*>(PyByteArray_AS_STRING(values));
  Py_BEGIN_ALLOW_THREADS;
  parallel(workers - 1, order, [&](size_t index) {
    Result& result = results[index];
    if (!result.positions.empty()) {
      std::memcpy(position_bytes + starts[index] * sizeof(uint32_t), result.positions.data(),
                  result.positions.size() * sizeof(uint32_t));
      std::memcpy(value_bytes + starts[index], result.values.data(), result.values.size());
    }
    result.positions = {};
    result.values = {};
  });
  Py_END_ALLOW_THREADS;
  std::string statuses(count, '\0');
  std::vector<uint32_t> checksums(count);
  PyObject* errors = PyDict_New();
  for (size_t index = 0; errors != nullptr && index < count; ++index) {
    statuses[index] = char(results[index].status);
    checksums[index] = results[index].checksum;
    if (results[index].status == kError) {
      PyObject* key = PyLong_FromSize_t(index);
      PyObject* message = PyUnicode_FromString(results[index].error.c_str());
      if (key == nullptr || message == nullptr || PyDict_SetItem(errors, key, message) != 0) {
        Py_CLEAR(errors);
      }
      Py_XDECREF(key);
      Py_XDECREF(message);
    }
  }
  if (errors == nullptr) {
    Py_DECREF(positions);
    Py_DECREF(values);
    return nullptr;
  }
  return Py_BuildValue("(y#y#y#NNN)", statuses.data(), Py_ssize_t(count),
                       reinterpret_cast<const char*>(checksums.data()), Py_ssize_t(count * sizeof(uint32_t)),
                       reinterpret_cast<const char*>(counts.data()), Py_ssize_t(count * sizeof(int64_t)),
                       positions, values, errors);
}

PyMethodDef methods[] = {
    {"check", py_check, METH_VARARGS,
     "check(paths, files, offsets, sizes, frames, bases, expected, overlays, threads)\n\n"
     "Check zstd XOR deltas against checkpoint files. Returns each tensor's status\n"
     "and target checksum, the counts of their nonzero delta bytes, those bytes'\n"
     "offsets and values concatenated, and the errors of malformed deltas."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_delta_ext", nullptr, -1, methods,
                      nullptr, nullptr, nullptr, nullptr};

}  // namespace

PyMODINIT_FUNC PyInit__delta_ext(void) { return PyModule_Create(&module); }
