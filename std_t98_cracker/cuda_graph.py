"""Optional fixed-batch CUDA replay for the unchanged FP32 hybrid model."""


class GraphRunner:
    def __init__(self, model, device, batch_size=1024):
        import torch
        if device.type != 'cuda':
            raise ValueError('CUDA graphs require a CUDA device')
        self.t = torch
        self.model, self.device = model, device
        self.sizes = sorted({min(n, batch_size) for n in (16, 64, 256, 1024)} | {batch_size})
        self.cache = {}
        # Replays are serialized and outputs are copied before reuse. Sharing
        # the graph pool avoids keeping a separate cuDNN workspace per bucket.
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream(device=device)

    def capture(self, size):
        t = self.t
        inputs = t.zeros((size, 20, 49), dtype=t.float32, device=self.device)
        stream = self.stream
        stream.wait_stream(t.cuda.current_stream(self.device))
        with t.cuda.stream(stream), t.inference_mode():
            for _ in range(3):
                self.model(inputs)
        t.cuda.current_stream(self.device).wait_stream(stream)
        graph = t.cuda.CUDAGraph()
        with t.inference_mode(), t.cuda.graph(graph, stream=stream, pool=self.pool):
            outputs = self.model(inputs)
        self.cache[size] = graph, inputs, outputs

    def warmup(self):
        for size in reversed(self.sizes):
            if size not in self.cache:
                self.capture(size)

    def __call__(self, inputs):
        count = len(inputs)
        if count == 0 or count > self.sizes[-1]:
            raise ValueError('graph input batch is empty or too large')
        size = next(n for n in self.sizes if n >= count)
        if size not in self.cache:
            self.capture(size)
        graph, source, output = self.cache[size]
        source[:count].copy_(inputs)
        graph.replay()
        # The next replay overwrites its output buffer. Return owned storage.
        return output[:count].clone()
