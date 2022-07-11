import os
import imp


def make_renderer(cfg, network, network_upper=None, network_lower=None):
    module = cfg.renderer_module
    path = cfg.renderer_path
    renderer = imp.load_source(module, path).Renderer(network, network_upper, network_lower)
    return renderer