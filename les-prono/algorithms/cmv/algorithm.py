import datetime as dt
import glob
import logging
import json
import os
from pathlib import Path
import pickle
from tqdm import tqdm

from dateutil.parser import isoparse
import cv2 as cv
import numpy as np
import pandas as pd

import algorithms.useful_functions as uf



def predict(loc, dcfg):
    """Predicts rp value for location by computing
    a cloud motion vector field and projecting
    accordingly.

    Args:
        loc (dict): [description]
        dcfg (dict): [description]

    Returns:
        datetime.datetime: datetime of the forecast moment
    """
    cmv_path = os.path.join(  # get cloud motion velocity field numpy file path
        os.path.dirname(os.path.realpath(__file__)), "cmv.npy"
    )
    (
        imgs_list,
        last_img_Path,
        last_dtime,
        created_dtime,
    ) = uf.analyze_psat(dcfg, True)  # get info from psat folder
    logging.debug("Check last img dtime")
    period = uf.get_last_period(imgs_list)  # difference between last imgs dtimes, used to project
    logging.debug("Get last time difference")
    aux_variables, now = load_variables()  # load last run time and get current time
    logging.debug("Load aux variables")
    if aux_variables["last_run_dtime"] < created_dtime:  # check if there's a new image
        cmv = get_cmv(dcfg, imgs_list)  # compute new cloud motion vector field
        np.save(cmv_path, cmv)  # save the cmv to a .npy file
        logging.debug("Compute new vector field")
        # make new projections
        update_predicted_imgs(
            last_img_Path, cmv, dcfg, overwrite=True, now=now
        )
        df2 = update_forecast(loc, dcfg)
        logging.debug("Make projections and create df2")
        # save forecasts to history
        hist_cmv_path, hist_df2_path = get_history_file_paths(
            dcfg, now, loc
        )
        df2.to_csv(hist_df2_path)
        logging.debug("Save to history")
    else:
        cmv = np.load(cmv_path)
        update_predicted_imgs(
            last_img_Path, cmv, dcfg, overwrite=False, now=now
        )
        df2 = update_forecast(loc, dcfg)
        logging.debug("Make lacking projections and create df2")
    logging.debug("Finish predict")
    return now


def create_proj_dir(dcfg):
    """Creates directory in which projections will be saved."""
    # input('create dir? (just checking)')
    proj_dir_path = Path(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "predictions",
            "projections",
        )
    )
    (proj_dir_path / "meta").mkdir(parents=True, exist_ok=True)
    if not os.path.isfile(proj_dir_path / "meta" / "lats.npy"):
        np.save(
            proj_dir_path / "meta" / "lats.npy",
            np.load(
                os.path.join(
                    dcfg["paths"]["psat"], "meta", "lats.npy"
                )
            ),
        )
        np.save(
            proj_dir_path / "meta" / "lons.npy",
            np.load(
                os.path.join(
                    dcfg["paths"]["psat"], "meta", "lons.npy"
                )
            ),
        )
    logging.debug("Ensure projections dir is there")
    return proj_dir_path


def update_predicted_imgs(
    base_img_Path, cmv, dcfg, overwrite, now=None):
    """Creates projections recursively. A lot of extra documentation is in the code."""
    # create dir and remove old projections if needed
    proj_dir_path = create_proj_dir(dcfg)
    if overwrite:  # clean target dir
        logging.debug("Overwriting...")
        for f in glob.glob(str(proj_dir_path / "*.npy")):
            os.remove(f)
    else:
        logging.debug("Avoiding projection removal")
    """Set current datetime, base image datetime,
    a placeholder for target datetimes (<idx>, 
    starting in the past), the desired target
    datetimes (<deltas_t_indices>) and their time
    difference with <base_img_datetime> (<deltas_t>).
    Also, intermediate timesteps are defined from
    <base_img_dtime> to 20 hours after that moment.
    Returns None if nothing to predict. base_img_Path is immutable.
    """
    dtime_now = (
        dt.datetime.now(dt.timezone.utc) if now is None else now
    )
    base_img_dtime = uf.get_dtime(base_img_Path.name)
    idx = uf.create_df2_datetimeindex(
        base_img_dtime,
        dtime_now,  # from moment previous to base img to limit hours after now
        dcfg["forecasts"]["time_step"],
        dcfg["forecasts"]["limit"],
    )
    logging.debug("Create datetimeindex of target df2 from now on")
    """Predicts an RP image from [x1name, x2name, ...] and [y1name, y2name, ...]"""
    deltas_t_indices = [
        ind
        for ind in idx  
        if overwrite
        or not os.path.exists(proj_dir_path / f"proj_{ind}.npy")
    ]  # select target datetimes if nonexistent or <overwrite == True>
    deltas_t = [
        (ind - base_img_dtime).total_seconds()
        for ind in idx  # first one is negative (it is before base img)
        if overwrite
        or not os.path.exists(proj_dir_path / f"proj_{ind}.npy")
    ]
    if len(deltas_t) <= 1:
        return None
    timesteps = pd.date_range(  # timesteps from base image and on
        start=base_img_dtime,
        end=base_img_dtime + dt.timedelta(hours=20),
        freq=f"{dcfg['algorithm']['cmv']['timestep']}min",  # beware of what happens if exceeds 60
        closed=None,  # closed at both sides
    )  # compute timesteps till 20 hours ahead

    """Initialization of variables. <base_img> is
    loaded, <base_dtime> is set to the first
    timestep (equal to <base_img_dtime>). The
    first forecasted image is defined as <base_img>,
    <current_index> to predict is set to 1, along
    with its corresponding time difference 
    (<delta_t>). <break_for_loop> flag is also set. """
    # init variables
    base_img = np.load(base_img_Path)
    base_dtime = timesteps[0]
    fimgs = [
        base_img,
    ]
    current_index = 1
    delta_t = deltas_t[current_index]
    break_for_loop = False
    print("projecting...")
    """Projection of images. We advance predicting
    the image corresponding to the next timestep.
    If we have a desired <delta_t> before the
    following timestep, then we forecast an image
    for that <delta_t> and append it to the list 
    of desired projections (<fimgs>). For the 
    forecast we use as base image the last image 
    projected (corresponding to <timesteps[i-1]>). 
    We do this as many times as needed (inside the
    while loop) and once ended we compute the 
    image corresponding to the next timestep.
    When all desired deltas_t have a corresponding
    image we exit the for loop and save the 
    projections.
    """
    # import pdb; pdb.set_trace()
    # advancing in timesteps
    for i in range(1, len(timesteps)):
        # make projection to delta t if flag or to timestep if not flag
        fimg, next_flag = compare_and_compute(
            base_dtime,
            base_img,
            delta_t,
            timesteps[i - 1],
            timesteps[i],
            cmv,
        )
        while next_flag:  # if we have a new image then append it
            fimgs.append(fimg)
            # advance to the next desired image and check
            current_index += 1
            if current_index == len(deltas_t):
                break_for_loop = True
                break
            else:  # compute fimg again
                delta_t = deltas_t[current_index]
                fimg, next_flag = compare_and_compute(
                    base_dtime,
                    base_img,
                    delta_t,
                    timesteps[i - 1],
                    timesteps[i],
                    cmv,
                )
        if break_for_loop:
            break
        base_img = fimg
    print("end projecting")
    logging.debug(
        f"len(dt indices) = {len(deltas_t_indices)}, len(fimgs) = {len(fimgs)}"
    )
    for m in range(len(deltas_t_indices)):
        new_img_Path = (
            proj_dir_path / f"proj_{deltas_t_indices[m]}.npy"
        )
        np.save(new_img_Path, fimgs[m])



def compare_and_compute(last_img_dtime, base_img, delta_t, from_dt, to_dt, cmf):
    """ Compares desired delta_t prediction with the datetimes of the current and next steps (from_dt and to_dt).
    Returns flag in True if there is an image to be computed between these two timesteps.

    Arguments:
        base_img {numpy.ndarray} -- base normalized image to project
        delta_t {float} -- time horizon from imgf datetime in seconds
        from_dt {datetime.datetime} -- base image datetime
        to_dt {datetime.datetime} -- next base image datetime

    Returns:
        numpy.ndarray -- forecasted image
        numpy.ndarray -- blurred forecasted image
        bool -- flag (True if from_dt < delta_t + self._last_img_dtime < to_dt)
    """
    total_delta_t = (
        to_dt - last_img_dtime
    ).total_seconds()  # total lt at this step
    if delta_t < total_delta_t:  # if exceeds desired delta_t
        flag = True
        dt = (
            delta_t - (from_dt - last_img_dtime).total_seconds()
        )  # set delta-time from base image datetime
    else:
        flag = False
        dt = (to_dt - from_dt).total_seconds()
    fimg = project_cmv(cmf, base_img, dt, show_for_debugging=False)
    # fimg = restore(fimg)
    return fimg, flag  # bimg, flag


def restore(img):
    """ Not implemented for now. Placeholder for any postprocessing over the projections.

    Args:
        img (numpy.ndarray): any image

    Returns:
        numpy.ndarray: processed image
    """
    pass
    return img


def project_cmv(cmv, base_img, delta_t, show_for_debugging=False):
    """ Projects <base_img> <delta_t> seconds in the future using the provided <cmv>.
    It's the simplest projection. Interpolation is linear, borders come into the image as nans.

    Args:
        cmv (numpy.ndarray): cloud motion vector field. Two channel image.
        base_img (numpy.ndarray): first or initial image
        delta_t (float): lead time in seconds
        show_for_debugging (bool, optional): only for debugging purposes. Defaults to False.

    Returns:
        numpy.ndarray: projected image
    """
    map_x, map_y = get_mapping(cmv, delta_t)
    next_img = cv.remap(
        base_img,
        map_x,
        map_y,
        cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    if show_for_debugging:
        show_for_debugging2(base_img, next_img, cmv, delta_t)
    return next_img


def get_mapping(cmv, delta_t):
    """Gets mapping needed for projection (remap).

    Args:
        cmv (numpy.ndarray): cloud motion vector field, two channel image
        delta_t (float): total seconds

    Returns:
        numpy.ndarray: Images with each pixel containing the destination x (y) coordinate.
    """
    i_idx, j_idx = np.meshgrid(
        np.arange(cmv.shape[1]), np.arange(cmv.shape[0])
    )
    map_i = i_idx + cmv[:, :, 0] * delta_t
    map_j = j_idx + cmv[:, :, 1] * delta_t
    return map_i.astype(np.float32), map_j.astype(np.float32)


def get_cmv(dcfg, imgs_list, show_for_debugging=False):
    """Creates cloud motion vector field from last two images"""
    imgs_paths = [
        Path(os.path.join(dcfg["paths"]["psat"], imgs_list[i]))
        for i in [-1, -2]
    ]
    imgf, imgi = np.load(imgs_paths[0]), np.load(imgs_paths[1])
    period = uf.get_last_period(sorted([img.name for img in imgs_paths]))
    logging.debug("Load imgs in t and t-1")
    cmvcfg = dcfg["algorithm"]["cmv"]
    flow = cv.calcOpticalFlowFarneback(
        imgi,
        imgf,
        None,
        pyr_scale=cmvcfg["pyr_scale"],
        levels=cmvcfg["levels"],
        winsize=cmvcfg["winsize"],
        iterations=cmvcfg["iterations"],
        poly_n=cmvcfg["poly_n"],
        poly_sigma=cmvcfg["poly_sigma"],
        flags=0,
    )
    logging.debug("Calculate optical flow")
    cmv = - flow / period.total_seconds()
    logging.debug("Calculate vector field in pix/second")
    if show_for_debugging:
        show_for_debugging1(cmv, imgf, imgi)
    return cmv


def load_variables():
    """Loads json in this folder or creates it if 
    nonexistent. Also returns current time.

    Returns:
        dict: aux_variables dict
        datetime.datetime: current time
    """
    now = dt.datetime.now(dt.timezone.utc)
    aux_variables_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "aux_variables.json",
    )
    try:
        aux_variables = json.load(open(aux_variables_path, "r"))
        aux_variables["last_run_dtime"] = isoparse(
            aux_variables["last_run_dtime"]
        )
    except FileNotFoundError:
        logging.warning("There is no aux_variables.json")
        aux_variables = {
            "last_run_dtime": (now - dt.timedelta(days=1000))
        }  # initialize with a very distant datetime
    return aux_variables, now


def save_variables(aux_variables, now):
    aux_variables_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "aux_variables.json",
    )
    aux_variables["last_run_dtime"] = now
    json.dump(
        aux_variables, open(aux_variables_path, "w"), default=str
    )


def update_forecast(loc, dcfg):
    """Update forecasts file up to limit hours using the projections"""
    proj_dir_path = Path(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "predictions",
            "projections",
        )
    )
    idx = create_datetimeindex_from_projections(proj_dir_path)
    logging.debug("Create datetimeindex array")
    df2 = pd.DataFrame(
        index=idx, columns=["predictions", "inf", "sup",]
    )
    for img_dtime in idx:
        df2["predictions"][img_dtime] = uf.get_value_gaussian(
            loc,
            dcfg["algorithm"]["cmv"]["kernel"],
            proj_dir_path / f"proj_{img_dtime}.npy",
        )
        df2["inf"][img_dtime] = 687  # placeholder for confidence interval, only alabel for now
        if df2["predictions"][img_dtime] <= 0:
            logging.warning(f"Value from projection is not positive, it's {df2['predictions'][img_dtime]}")
    logging.debug("Create df2")
    uf.write_to_daydf(df2, proj_dir_path.parent / loc["name"], loc)
    logging.debug("Update daydf")
    return df2


def create_datetimeindex_from_projections(proj_dir_path):
    """Loads projections and creates a DatetimeIndex according to their name.

    Args:
        proj_dir_path (str): path to the folder containing the projections

    Returns:
        pandas.DatetimeIndex: indices with datetimes corresponding to projections
    """
    idx = []
    imgs_list = [
        name
        for name in os.listdir(proj_dir_path)
        if not os.path.isdir(os.path.join(proj_dir_path, name))
    ]
    for img_name in imgs_list:
        img_dtime = dt.datetime.strptime(
            Path(img_name).stem.split("roj_")[1][:19],  # ignore loc
            "%Y-%m-%d %H:%M:%S",
        )
        idx.append(img_dtime)
    idx = pd.DatetimeIndex(sorted(idx))
    idx = idx.tz_localize(dt.timezone.utc)
    idx.freq = pd.infer_freq(idx)
    return idx


def get_history_file_paths(dcfg, now=None, loc=None):
    """Gets file paths for cmv.npy and df2.csv"""
    now = dt.datetime.now(dt.timezone.utc) if now is None else now

    history_file_path_cmv = Path(
        os.path.join(
            dcfg["paths"]["history"],
            "cmv",
            loc['name'],
            str(now.date()),
            f"cmv_{now}.npy",
        )
    )
    history_file_path_df2 = str(
        history_file_path_cmv.parent / f"{now}.csv"
    ).replace(" ", "_")
    logging.debug("Get history paths")

    history_file_path_cmv.parent.mkdir(parents=True, exist_ok=True)
    logging.debug("Create history directory")

    return history_file_path_cmv, history_file_path_df2


def show_for_debugging1(cmv, imgf, imgi):
    """For debugging purposes. Show cmv as color image.
    """
    flow = cmv
    print(cmv.shape)
    import matplotlib.pyplot as plt

    hsv = np.zeros((imgf.shape[0], imgf.shape[1], 3), np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    plt.figure()
    plt.imshow(imgi)
    plt.figure()
    plt.imshow(imgf)
    plt.figure()
    plt.imshow(bgr)
    plt.show()


def show_for_debugging2(base_img, next_img, cmv, delta_t):
    """For debugging purposes. Saves first and projected images.
    """
    import matplotlib.pyplot as plt
    print("minutes", delta_t / 60)
    print(np.amax(cmv[:, :, 0]), np.amin(cmv[:, :, 0]))
    print(np.amax(cmv[:, :, 1]), np.amin(cmv[:, :, 1]))
    plt.figure()
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    plt.title("base")
    plt.imshow(base_img)
    plt.savefig("testing/cmv/base")
    plt.figure()
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    plt.title("next")
    plt.imshow(next_img)
    plt.savefig("testing/cmv/next")
    # input("Did you see the imgs?")
    # plt.show()

