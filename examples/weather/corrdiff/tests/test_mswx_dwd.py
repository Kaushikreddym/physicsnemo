import importlib.util, sys, os

datasets_dir = "/beegfs/muduchuru/codes/physicsnemo/examples/weather/corrdiff/datasets"

# Create a package module object for 'datasets'
spec_pkg = importlib.util.spec_from_loader("datasets", loader=None)
pkg = importlib.util.module_from_spec(spec_pkg)
pkg.__path__ = [datasets_dir]
sys.modules["datasets"] = pkg

# Now load the submodule as 'datasets.mswx_dwd'
spec = importlib.util.spec_from_file_location("datasets.mswxdwd",
                                              os.path.join(datasets_dir, "mswxdwd.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

print(mod.mswxdwd)  # class is available

dataloader = mod.mswxdwd(
    data_path='/beegfs/muduchuru/data',
    input_channels = ['pr','tas'],
    output_channels = ['pr','tas'],
    static_channels = ['elevation', 'lsm','dwd_mask','pos_embed'],
    stats_dwd = '/beegfs/muduchuru/data/hyras_daily/hyras_stats.json',
    stats_mswx = '/beegfs/muduchuru/data/mswx/mswx_stats.json',
    )
data = dataloader[0]