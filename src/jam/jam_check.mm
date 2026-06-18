#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "src/jam/jam.h"
#include "src/jam/program_impl.h"

namespace mccl_jax::jam { std::vector<std::string> RegisteredOpNames(); }

namespace {

std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

struct DTInfo { size_t bytes; MPSDataType mps; };
DTInfo Info(mccl_jax::jam::DType d) {
  using DT = mccl_jax::jam::DType;
  switch (d) {
    case DT::kF16:  return {2, MPSDataTypeFloat16};
    case DT::kBF16: return {2, MPSDataTypeBFloat16};
    case DT::kI8:   return {1, MPSDataTypeInt8};
    case DT::kPred: return {1, MPSDataTypeBool};
    case DT::kI64:  return {4, MPSDataTypeInt32};
    case DT::kI32:  return {4, MPSDataTypeInt32};
    default:        return {4, MPSDataTypeFloat32};
  }
}

size_t NumElems(const std::vector<int64_t>& dims) {
  size_t n = 1;
  for (int64_t d : dims) n *= (size_t)(d < 0 ? 0 : d);
  return n;
}

NSArray<NSNumber*>* MpsShape(const std::vector<int64_t>& dims) {
  NSMutableArray<NSNumber*>* a = [NSMutableArray array];
  for (int64_t d : dims) [a addObject:@(d)];
  if (a.count == 0) [a addObject:@1];
  return a;
}

}

int main(int argc, const char** argv) {
  @autoreleasepool {

    if (argc >= 2 && std::string(argv[1]) == "--list-ops") {
      auto names = mccl_jax::jam::RegisteredOpNames();
      printf("jam registered op handlers: %zu\n", names.size());
      for (const auto& n : names) printf("  %s\n", n.c_str());
      return 0;
    }

    bool bench = (argc >= 2 && std::string(argv[1]) == "--bench");
    int bench_n = 50;
    int argbase = 1;
    if (bench) {
      if (argc < 4) { fprintf(stderr, "usage: %s --bench <N> <artifact> [inputs...]\n", argv[0]); return 2; }
      bench_n = atoi(argv[2]);
      if (bench_n < 1) bench_n = 1;
      argbase = 3;
    }
    if (!bench && argc < 3) {
      fprintf(stderr, "usage: %s <artifact.mlirbc> <out_dir> [in0.bin ...]\n", argv[0]);
      fprintf(stderr, "       %s --list-ops\n", argv[0]);
      fprintf(stderr, "       %s --bench <N> <artifact.mlirbc> [in0.bin ...]\n", argv[0]);
      return 2;
    }
    std::string artifact = ReadFile(argv[argbase]);
    std::string out_dir = bench ? "" : argv[2];
    int in_base = bench ? (argbase + 1) : 3;
    if (artifact.empty()) { fprintf(stderr, "jam_check: empty/unreadable artifact\n"); return 2; }

    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();
    auto result = mccl_jax::jam::Compile(artifact.data(), artifact.size());
    double compile_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    if (!result.program) {
      fprintf(stderr, "jam_check: Compile failed: %s\n", result.error.c_str());
      return 3;
    }
    auto* impl = result.program->impl();
    const auto& in_specs = result.program->inputs();
    const auto& out_specs = result.program->outputs();

    int n_in = argc - in_base;
    if (n_in != (int)in_specs.size()) {
      fprintf(stderr, "jam_check: program wants %zu inputs, got %d\n", in_specs.size(), n_in);
      return 4;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) { fprintf(stderr, "jam_check: no Metal device\n"); return 5; }
    id<MTLCommandQueue> queue = [device newCommandQueue];

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [NSMutableDictionary dictionary];
    for (int i = 0; i < n_in; ++i) {
      const auto& spec = in_specs[i];
      DTInfo dt = Info(spec.dtype);
      std::string raw = ReadFile(argv[in_base + i]);
      size_t want = NumElems(spec.dims) * dt.bytes;
      if (raw.size() != want) {
        fprintf(stderr, "jam_check: input %d: got %zu bytes want %zu\n", i, raw.size(), want);
        return 6;
      }
      id<MTLBuffer> buf = [device newBufferWithBytes:raw.data()
                                              length:(raw.empty() ? 1 : raw.size())
                                             options:MTLResourceStorageModeShared];
      MPSGraphTensorData* td = [[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                                      shape:MpsShape(spec.dims)
                                                                   dataType:dt.mps];
      feeds[impl->inputs[i]] = td;
    }

    NSMutableArray<MPSGraphTensor*>* targets = [NSMutableArray array];
    for (MPSGraphTensor* t : impl->outputs) [targets addObject:t];

    if (bench) {

      for (int w = 0; w < 3; ++w) {
        @autoreleasepool {
          (void)[impl->graph runWithMTLCommandQueue:queue feeds:feeds targetTensors:targets targetOperations:nil];
        }
      }
      double total = 0, best = 1e30;
      for (int it = 0; it < bench_n; ++it) {
        @autoreleasepool {
          auto s = Clock::now();
          MPSGraphTensorDataDictionary* r = [impl->graph runWithMTLCommandQueue:queue feeds:feeds
                                                                  targetTensors:targets targetOperations:nil];

          MPSGraphTensorData* td = r[(MPSGraphTensor*)targets[0]];
          (void)[td mpsndarray];
          double ms = std::chrono::duration<double, std::milli>(Clock::now() - s).count();
          total += ms; if (ms < best) best = ms;
        }
      }
      printf("BENCH compile_ms=%.3f gpu_mean_ms=%.4f gpu_min_ms=%.4f runs=%d\n",
             compile_ms, total / bench_n, best, bench_n);

      auto te0 = Clock::now();
      NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feedShapes =
          [NSMutableDictionary dictionary];
      for (int i = 0; i < n_in; ++i) {
        feedShapes[impl->inputs[i]] = [[MPSGraphShapedType alloc]
            initWithShape:MpsShape(in_specs[i].dims) dataType:Info(in_specs[i].dtype).mps];
      }
      MPSGraphCompilationDescriptor* cd = [[MPSGraphCompilationDescriptor alloc] init];
      MPSGraphExecutable* exe = [impl->graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                                         feeds:feedShapes
                                                 targetTensors:targets
                                              targetOperations:nil
                                         compilationDescriptor:cd];
      double exe_compile_ms = std::chrono::duration<double, std::milli>(Clock::now() - te0).count();
      if (exe != nil) {

        NSArray<MPSGraphTensor*>* feedOrder = exe.feedTensors;
        NSMutableArray<MPSGraphTensorData*>* inputsArray = [NSMutableArray array];
        for (MPSGraphTensor* t : feedOrder) [inputsArray addObject:feeds[t]];
        MPSGraphExecutableExecutionDescriptor* ed =
            [[MPSGraphExecutableExecutionDescriptor alloc] init];
        for (int w = 0; w < 3; ++w) {
          @autoreleasepool {
            (void)[exe runWithMTLCommandQueue:queue inputsArray:inputsArray
                                 resultsArray:nil executionDescriptor:ed];
          }
        }
        double etotal = 0, ebest = 1e30;
        for (int it = 0; it < bench_n; ++it) {
          @autoreleasepool {
            auto s = Clock::now();
            NSArray<MPSGraphTensorData*>* res = [exe runWithMTLCommandQueue:queue inputsArray:inputsArray
                                                              resultsArray:nil executionDescriptor:ed];
            (void)[res[0] mpsndarray];
            double ms = std::chrono::duration<double, std::milli>(Clock::now() - s).count();
            etotal += ms; if (ms < ebest) ebest = ms;
          }
        }
        printf("BENCH_EXE compile_ms=%.3f gpu_mean_ms=%.4f gpu_min_ms=%.4f runs=%d\n",
               exe_compile_ms, etotal / bench_n, ebest, bench_n);
      } else {
        printf("BENCH_EXE compile failed (exe==nil)\n");
      }
      return 0;
    }

    MPSGraphTensorDataDictionary* results = [impl->graph runWithMTLCommandQueue:queue
                                                                          feeds:feeds
                                                                  targetTensors:targets
                                                               targetOperations:nil];

    for (size_t i = 0; i < out_specs.size(); ++i) {
      const auto& spec = out_specs[i];
      DTInfo dt = Info(spec.dtype);
      size_t n = NumElems(spec.dims);
      std::vector<uint8_t> buf(n * dt.bytes);
      MPSGraphTensorData* td = results[(MPSGraphTensor*)targets[i]];
      MPSNDArray* nda = [td mpsndarray];
      [nda readBytes:buf.data() strideBytes:nil];

      std::ostringstream path;
      path << out_dir << "/out" << i << ".bin";
      std::ofstream f(path.str(), std::ios::binary);
      f.write((const char*)buf.data(), (std::streamsize)buf.size());
    }

    fprintf(stderr, "jam_check: ran %zu inputs -> %zu outputs\n", in_specs.size(), out_specs.size());
  }
  return 0;
}
