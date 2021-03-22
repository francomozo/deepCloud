import datetime
import datetime as dt
import os
from pathlib import Path
import sys

sys.path.append(os.path.abspath("algorithms"))

import numpy as np
import pandas as pd

import useful_functions as uf

# import useful_functions as uf


def test_get_dtime():
    img_name = "ART_2020359_134300_rp.npy"
    dtime = uf.get_dtime(img_name)
    assert dtime == (
        datetime.datetime(
            year=2020,
            month=1,
            day=1,
            hour=13,
            minute=43,
            second=0,
            tzinfo=dt.timezone.utc,
        )
        + datetime.timedelta(days=358)
    )


def test_get_consistent_period():
    imgs_list = [
        "ART_2020359_134300_rp.npy",
        "ART_2020359_135300_rp.npy",
        "ART_2020359_140300_rp.npy",
        "ART_2020359_132300_rp.npy",
    ]
    assert uf.get_consistent_period(imgs_list) == datetime.timedelta(
        minutes=10
    )


def test_get_needed_lead_times():
    limit = 4
    period = datetime.timedelta(minutes=10)
    assert uf.get_needed_lead_times(limit, period) == 24


def test_get_pixel():
    loc = {"lat": -4.3, "lon": 2.1}
    img = (-np.arange(10)[:, np.newaxis]) * np.arange(5)[
        np.newaxis, :
    ]
    lats, lons = -np.arange(10), np.arange(5)
    img_path = "project/testing/test_img.npy"
    np.save(img_path, img)

    target = uf.get_pixel(loc, img_path, lats=lats, lons=lons)
    os.remove(img_path)
    assert target == -8


def test_get_value():
    loc = {"lat": -4.3, "lon": 2.1}
    img = (-np.arange(10)[:, np.newaxis]) * np.arange(5)[
        np.newaxis, :
    ]
    lats, lons = -np.arange(10), np.arange(5)
    img_path = "project/testing/test_img.npy"
    np.save(img_path, img)
    winsize = 2  # should be the same for winsize=3
    target = uf.get_value(
        loc, winsize, img_path, lats=lats, lons=lons
    )
    os.remove(img_path)
    assert target == -8


def test_load_pandas_dataframe():
    idx = pd.date_range(
        start=datetime.datetime.now(),
        end=(datetime.datetime.now() + datetime.timedelta(hours=3)),
        freq="10min",
    )
    a = pd.DataFrame(
        np.arange(2 * len(idx)).reshape((len(idx), 2)),
        index=idx,
        columns=["first", "2"],
    )
    a.to_csv("test_df")
    b = uf.load_pandas_dataframe("test_df")
    os.remove("test_df")
    assert np.all(b == a)


def test_next_step_starts():
    dtime = datetime.datetime(
        year=2020, month=12, day=31, hour=23, minute=50
    )
    time_step = 12
    nss = uf.next_step_starts(dtime, time_step)
    assert nss == datetime.datetime(2021, 1, 1)


def test_previous_step_starts():
    dtime = datetime.datetime(
        year=2020, month=1, day=1, hour=0, minute=0
    )
    time_step = 12
    pss = uf.previous_step_starts(dtime, time_step)
    assert pss == datetime.datetime(2020, 1, 1)


def test_interpolate_behavior():
    idx = pd.date_range(
        start=datetime.datetime.now(),
        end=(datetime.datetime.now() + datetime.timedelta(hours=3)),
        freq="10min",
    )
    idx2 = pd.date_range(
        start=(
            datetime.datetime.now() - datetime.timedelta(minutes=33)
        ),
        end=(datetime.datetime.now() + datetime.timedelta(hours=3)),
        freq="10min",
    )

    a = pd.DataFrame(
        np.arange(len(idx)),
        index=idx,
        columns=["first"],
        dtype=np.float,
    )
    b = pd.DataFrame(index=idx2, columns=["first"])
    # print(a.dtypes)
    # print(b)
    df2 = pd.concat([a, b]).sort_index().interpolate("time")
    df2[df2.index < a.index[0]] = a.iloc[0, 0]
    # print(df2)


def test_df2_to_daydf(verbose=False):
    today = dt.datetime.now(dt.timezone.utc).date()
    daydf = uf.create_daydf(
        today,
        dt.timedelta(minutes=30),
        columns=["predictions", "inf"],
        tzone="America/Montevideo",
    )
    df2_index = pd.date_range(
        uf.date_to_datetime(today) + dt.timedelta(hours=12),
        uf.date_to_datetime(today) + dt.timedelta(hours=17),
        freq=dt.timedelta(minutes=30),
        tz=dt.timezone.utc,
    )
    df2 = pd.DataFrame(
        3 * np.ones(len(df2_index)),
        index=df2_index,
        columns=["predictions"],
    )
    if verbose:
        print(df2)
        print(uf.df2_to_daydf(df2, daydf))


def test_today_plus_hours():
    a = dt.datetime.now(dt.timezone.utc).date()
    b = uf.date_to_datetime(a)
    c = b + dt.timedelta(hours=13)
    assert c == dt.datetime(a.year, a.month, a.day, hour=13)


def test_timezone_convert(verbose=False):
    idx = pd.date_range(
        dt.datetime.now(),
        dt.datetime.now() + dt.timedelta(hours=3),
        freq=dt.timedelta(minutes=30),
        tz="America/Montevideo",  # dt.timezone.utc
    )
    df2 = pd.DataFrame(np.ones(len(idx)), index=idx)
    loc = {"lat": -15.18, "lon": -60}
    new_df2, tzone = uf.timezone_convert(df2, loc)
    if verbose:
        print(df2)
        print(new_df2)
        print(tzone)
        print(type(tzone))


def save_cmv_projections():
    dir_path = os.path.join(
        "algorithms", "cmv", "predictions", "projections"
    )

    proj_names = [
        name for name in os.listdir(dir_path) if name != "meta"
    ]

    import matplotlib.pyplot as plt

    Path("algorithms/testing_imgs").mkdir(
        parents=True, exist_ok=True
    )
    for proj_name in proj_names[:10]:
        img = np.load(os.path.join(dir_path, proj_name))
        plt.figure()
        plt.title(proj_name)
        plt.imshow(img)
        plt.savefig(f"algorithms/testing_imgs/{proj_name}"[:-4])


def test_normalization(verbose=False):
    a = np.random.randint(0, 10, (5, 4)).astype(np.float)
    cosangs = np.random.randn(5, 4)

    if verbose:
        print("A", a)
        print("cos", cosangs)

    b = np.divide(
        a, cosangs, out=np.zeros_like(a), where=cosangs > 0.5
    )
    c = np.divide(
        a,
        cosangs,
        out=np.zeros_like(a),
        where=(0 < cosangs) * (cosangs <= 0.5),
    )
    d = b + np.clip(c, a_min=0, a_max=np.amax(b))

    if verbose:
        print("b", b)
        print("c", c)
        print("d", d)


def clean_interpolate():
    values, placeholder = create_example_dataframes()
    print("------------")
    print("values")
    print(values)
    print("------------")
    print("placeholder")
    print(placeholder)

    for col in placeholder.columns:
        placeholder[col].loc[:] = np.nan
    print("------------")
    print("cleaned placeholder")
    print(placeholder)

    placeholder.update(values)
    print("------------")
    print("updated placeholder")
    print(placeholder)

    placeholder = pd.concat([placeholder, values], axis=0)
    print("------------")
    print("concatenated placeholder")
    print(placeholder)

    placeholder = placeholder.loc[
        ~placeholder.index.duplicated(keep="first")
    ]
    print("------------")
    print("without duplicates placeholder")
    print(placeholder)

    placeholder = placeholder.sort_index()
    print("------------")
    print("sorted placeholder")
    print(placeholder)

    placeholder = placeholder.interpolate(
        "time", limit_area="inside"
    )
    print("------------")
    print("interpolated placeholder")
    print(placeholder)


def create_example_dataframes():
    # import pdb; pdb.set_trace()
    idxa = pd.date_range(
        dt.datetime.now(dt.timezone.utc),
        dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=5),
    )
    dfa = pd.DataFrame(
        np.arange(len(idxa)), index=idxa, columns=["value"]
    )
    idxb = idxa.shift(3).union([dt.datetime.now(dt.timezone.utc)])
    dfb = pd.DataFrame(
        -np.arange(len(idxb)), index=idxb, columns=["value"]
    )
    return dfa, dfb


'''Run this from above 'project' running
"python -m project.testing.test_useful_functions"'''
if __name__ == "__main__":
    # df1, df2 and daydf handling are still to be tested
    # cosangs related functions are still to be tested
    # clean_interpolate()

    # test_get_dtime()
    # test_get_consistent_period()
    # test_get_needed_lead_times()
    # test_get_pixel()
    # test_get_value()
    # test_load_pandas_dataframe()
    # test_next_step_starts()
    # test_previous_step_starts()
    # test_interpolate_behavior()
    # test_df2_to_daydf(verbose=False)
    # test_today_plus_hours()
    # test_timezone_convert(verbose=False)
    # # save_cmv_projections()
    # test_normalization(verbose=False)

    from CIM_module.funcion_CIMGHI_dummy import (
        generar_CIM_GHI_dumy as alf,
    )

    ts_out = uf.load_pandas_dataframe(
        os.path.join("predictions", "LE", "2020-06-24.csv")
    )
    loc = {"name": "LE", "lat": -30, "lon": -60}
    paso = round(pd.to_timedelta(ts_out.index.freq).total_seconds()/60)
    
    ghi = alf(
        ts_out.index.tz_convert("UTC"),
        ts_out.iloc[:, 0].values,
        loc["lat"],
        loc["lon"],
        paso,
        est=loc["name"],
    )
    out = pd.DataFrame(index = ts_out.index, data=ghi.values, columns=['ghi'])
    
    import matplotlib.pyplot as plt
    plt.plot(ts_out, label='rp')
    plt.plot(out, label='ghi')
    plt.legend()
    plt.savefig('ghi.png')
    import pdb; pdb.set_trace()

