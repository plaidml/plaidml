import time
import unittest

import plaidml2.bridge.pytorch as plaidml_pytorch
import torch


def printf(*args, **kwargs):
    print(*args, **kwargs, flush=True)


class TestBase(unittest.TestCase):

    def _run_jit(self, func, inputs):
        with torch.no_grad():
            trace_jit = torch.jit.trace(func, inputs)
            jit_out = trace_jit(*inputs)
            assert "plaidml::CompilationGroup" not in str(trace_jit.graph_for(*inputs))
            return jit_out

    def _run_pml(self, func, inputs):
        with torch.no_grad():
            with plaidml_pytorch.toggle():
                trace_pml = torch.jit.trace(func, inputs)
                pml_out = trace_pml(*inputs)
                assert "plaidml::CompilationGroup" in str(trace_pml.graph_for(*inputs))
        return pml_out

    def _run_both(self, func, inputs):
        return self._run_jit(func, inputs), self._run_pml(func, inputs)

    def _benchmark(self, func, inputs, iters=100, warmup=10):
        with torch.no_grad():
            printf("Tracing model with JIT")
            trace_jit = torch.jit.trace(func, inputs)
            printf("Warming JIT up with {} runs".format(warmup))
            for _ in range(warmup):
                _ = trace_jit(*inputs)

            printf("Running JIT {} times".format(iters))
            start = time.time()
            for _ in range(iters):
                _ = trace_jit(*inputs)
            jit_time = time.time() - start
            printf("Done benchmarking JIT")

            with plaidml_pytorch.toggle():
                printf("Tracing model with PlaidML")
                trace_pml = torch.jit.trace(func, inputs)
                printf("Warming PlaidML up with {} iters".format(warmup))
                for _ in range(warmup):
                    _ = trace_pml(*inputs)

                printf("Running PlaidML {} times".format(iters))
                start = time.time()
                for _ in range(iters):
                    _ = trace_pml(*inputs)
                pml_time = time.time() - start
                with torch.autograd.profiler.profile() as prof:
                    _ = trace_pml(*inputs)

            pml_profiled_time = 0
            total_profiled_time = 0
            for p in prof.key_averages():
                total_profiled_time += int(p.cpu_time)
                if p.key == "PlaidML":
                    pml_profiled_time += int(p.cpu_time)
            printf("Done benchmarking PlaidML, which compiled {:.2f}% of compute".format(
                100 * pml_profiled_time / total_profiled_time))

        printf("JIT: {} iter/s".format(iters / jit_time))
        printf("PML: {} iter/s".format(iters / pml_time))
