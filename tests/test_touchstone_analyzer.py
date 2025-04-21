import pytest
import pandas as pd
from pathlib import Path
import xarray as xr
import tempfile

from mwlab import TouchstoneDataset
from mwlab.transforms import s_transforms as st, TComposite
from mwlab.utils.analysis import TouchstoneDatasetAnalyzer


# ------------------ ФИКСТУРЫ ------------------

@pytest.fixture(scope="module")
def dataset_resampled():
    root = Path(__file__).parent.parent  / "Data" / "Filter12"
    s_tf = TComposite([
        st.S_Resample(freq_or_n=100)
    ])
    return TouchstoneDataset(root=root, s_tf=s_tf)

@pytest.fixture(scope="module")
def analyzer(dataset_resampled):
    return TouchstoneDatasetAnalyzer(dataset=dataset_resampled)


# ------------------ ПАРАМЕТРЫ ------------------
def test_dataset_has_parameters(analyzer):
    df = analyzer.get_params_df()
    assert df.shape[0] > 0, "Нет образцов"
    assert df.shape[1] > 0, "Нет параметров модели"

def test_get_params_df(analyzer):
    df = analyzer.get_params_df()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == len(analyzer.ds)

def test_summarize_params(analyzer):
    stats = analyzer.summarize_params()
    assert isinstance(stats, pd.DataFrame)
    assert "mean" in stats.index
    assert "is_constant" in stats.index

def test_get_varying_keys(analyzer):
    keys = analyzer.get_varying_keys()
    assert isinstance(keys, list)
    assert all(isinstance(k, str) for k in keys)

def test_plot_param_distributions(analyzer):
    fig = analyzer.plot_param_distributions()
    if fig is not None:
        assert hasattr(fig, "savefig")


# ------------------ S-МАТРИЦЫ ------------------

def test_assemble_s_xarray(analyzer):
    da = analyzer._assemble_s_xarray()
    assert isinstance(da, xr.DataArray)
    assert da.dims == ('sample', 'real_imag', 'freq', 'port_out', 'port_in')
    assert da.shape[0] == len(analyzer.ds)

def test_get_s_metric_db(analyzer):
    da = analyzer._get_s_metric('db')
    assert isinstance(da, xr.DataArray)
    assert da.dims == ('sample', 'freq', 'port_out', 'port_in')

def test_get_s_metric_deg(analyzer):
    da = analyzer._get_s_metric('deg')
    assert isinstance(da, xr.DataArray)
    assert da.dims == ('sample', 'freq', 'port_out', 'port_in')

def test_summarize_s(analyzer):
    ds = analyzer.summarize_s()
    assert set(ds.data_vars) == {'mean', 'std', 'min', 'max'}
    assert 'port_out' in ds.coords and 'port_in' in ds.coords

def test_get_s_normalization_stats(analyzer):
    stats = analyzer.get_s_normalization_stats()
    assert isinstance(stats, dict)
    for k, v in stats.items():
        assert isinstance(k, tuple)
        assert all(name in v for name in ['mean', 'std', 'min', 'max'])

def test_plot_s_stats(analyzer):
    fig = analyzer.plot_s_stats(port_out=0, port_in=0)
    assert hasattr(fig, "savefig")

# ------------------ exportS ------------------

def test_export_params_csv(analyzer):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "params.csv"
        analyzer.export_params_csv(path)
        assert path.exists()
        df = pd.read_csv(path)
        assert df.shape[0] == len(analyzer.ds)

def test_export_s_netcdf(analyzer):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "s_stats.nc"
        analyzer.export_s_netcdf(path)
        assert path.exists()
        # Читаем обратно через xarray
        ds = xr.open_dataset(path)
        assert set(ds.data_vars) == {"mean", "std", "min", "max"}
        ds.close()
