from enums import TemperatureModel, IncidentAngleModel, SingleDiodeMethod
from pandas import DataFrame, Index, MultiIndex, NaT, Series
from pvlib import pvsystem, temperature, iam, irradiance

from .effect_generator import compute_modifiers, get_effect_names

def _operate_effective_irradiance(module_params, env_params, solar_positions, iam_model) -> Series:
    """
    Calculates the effective irradiance reaching the solar cells by applying beam and diffuse
    Incidence Angle Modifiers (IAM) based on the specified physical model.

    :param module_params: Dictionary containing static physical attributes of the array.
                          Must include ``surface_azimuth``, ``surface_tilt``, and the specific
                          glass parameters required by the chosen IAM model (``n``, ``K``, ``L``
                          for PHYSICAL; ``b`` for ASHRAE; ``a_r`` for MARTIN_RUIZ).
    :param env_params: ``pd.DataFrame`` containing the Plane of Array (POA) irradiances. Must
                       include columns ``poa_direct``, ``poa_sky_diffuse``, and ``poa_ground_diffuse``.
    :param solar_positions: ``pd.DataFrame`` containing the time-series geometric position of the sun.
                            Must include columns ``solar_azimuth`` and ``solar_zenith``.
    :param iam_model: ``IncidentAngleModel`` enum specifying which IAM model to apply.
    :return: ``pd.Series`` representing the total effective irradiance (W/m^2) absorbed by the cells.
    """

    aoi = irradiance.aoi(
        surface_azimuth=module_params['surface_azimuth'],
        surface_tilt=module_params['surface_tilt'],
        solar_azimuth=solar_positions['solar_azimuth'],
        solar_zenith=solar_positions['solar_zenith'],
    )

    if iam_model is IncidentAngleModel.PHYSICAL:
        iam_beam = iam.physical(aoi, module_params['n'], module_params['K'], module_params['L'])
        iam_diffuse = iam.marion_diffuse('physical', module_params['surface_tilt'], n = module_params['n'],
                                         K = module_params['K'], L = module_params['L'])

    elif iam_model is IncidentAngleModel.ASHRAE:
        iam_beam = iam.ashrae(aoi, module_params['b'])
        iam_diffuse = iam.marion_diffuse('ashrae', module_params['surface_tilt'], b = module_params['b'])

    elif iam_model is IncidentAngleModel.MARTIN_RUIZ:
        iam_beam = iam.martin_ruiz(aoi, module_params['a_r'])
        iam_diffuse = iam.marion_diffuse('martin_ruiz', module_params['surface_tilt'], a_r = module_params['a_r'])

    else:
        raise ValueError('Invalid IAM Model selected')

    return (env_params['poa_direct'] * iam_beam +
            env_params['poa_sky_diffuse'] * iam_diffuse['sky'] +
            env_params['poa_ground_diffuse'] * iam_diffuse['ground'])

def _operate_cell_temperature(temp_model: TemperatureModel, env_params) -> Series:
    """
    Calculates the operating cell temperature based on ambient environmental conditions
    and the thermal characteristics of the module's mounting structure.

    :param temp_model: ``TemperatureModel`` enum specifying the thermal model to use
                       (e.g., SAPM or PVSyst variants).
    :param env_params: ``pd.DataFrame`` containing time-series environmental data. Must include
                       columns ``poa_global``, ``temp_air``, and ``wind_speed``.
    :return: ``pd.Series`` representing the estimated cell temperature in degrees Celsius.
    """

    if temp_model not in [TemperatureModel.PVSYST_INSULATED, TemperatureModel.PVSYST_SEMI_INTEGRATED,
                               TemperatureModel.PVSYST_FREESTANDING]:
        return temperature.sapm_cell(
            poa_global=env_params['poa_global'],
            temp_air=env_params["temp_air"],
            wind_speed=env_params["wind_speed"],
            **temp_model.value
        )

    else:
        return temperature.pvsyst_cell(
            poa_global=env_params['poa_global'],
            temp_air=env_params["temp_air"],
            wind_speed=env_params["wind_speed"],
            **temp_model.value
        )

def generate_data(module_params: dict, env_params: DataFrame, solar_positions: DataFrame, temp_model: TemperatureModel,
                iam_model:  IncidentAngleModel, num_panels: int, method: SingleDiodeMethod) -> tuple[DataFrame, DataFrame]:
    """
    Executes the complete California Energy Commission (CEC) single-diode pipeline across a fleet of panels.

    For each panel: runs ``calcparams_cec`` for the five reference parameters, applies the per-panel
    effect modifiers to the effective irradiance, cell temperature, and those parameters,
    then runs ``singlediode`` for the seven primary I-V points. Only the I-V points are kept in the
    observables output, the modified single-diode parameters are intermediate and discarded each iteration.
    Observables are pre-allocated as a long-format ``pd.DataFrame`` indexed by ``(panel_id, timestamp)``.
    Per-panel results are written into contiguous row blocks via ``iloc`` to avoid a final ``concat`` copy.
    Per-panel iteration is the cache-friendly call pattern at this project's scale.

    A sibling attribution ``DataFrame`` is built alongside the observables: one row per panel, one
    column per registered effect (named ``{effect_name}_onset``). Cells hold the timestamp at which
    the effect triggered on that panel, or ``pd.NaT`` if it never triggered within the simulated
    horizon. The attribution frame is small (one row per panel) and is written as a sidecar parquet
    next to the observables.

    :param module_params: dictionary of the physical and electrical parameters of the module. Must
                          include mounting geometry, IAM glass parameters, and the CEC reference
                          parameters (``alpha_sc``, ``a_ref``, ``I_L_ref``, ``I_o_ref``, ``R_sh_ref``,
                          ``R_s``, ``Adjust``).
    :param env_params: ``pd.DataFrame`` containing weather and Plane of Array (POA) irradiances.
    :param solar_positions: ``pd.DataFrame`` containing the time-series geometric solar positions.
    :param temp_model: ``TemperatureModel`` enum for the cell temperature estimation.
    :param iam_model: ``IncidentAngleModel`` enum for the effective irradiance estimation.
    :param num_panels: number of panels in the fleet to simulate.
    :param method: ``SingleDiodeMethod`` enum selecting the ``singlediode`` resolution method.
    :return: tuple ``(observables, attribution)``. ``observables`` is indexed by
             ``(panel_id, timestamp)`` with seven I-V point columns (``i_sc``, ``v_oc``, ``i_mp``,
             ``v_mp``, ``p_mp``, ``i_x``, ``i_xx``). ``attribution`` is indexed by ``panel_id``
             with one ``{effect_name}_onset`` column per registered effect.
    """

    effective_irradiance = _operate_effective_irradiance(module_params, env_params, solar_positions, iam_model)
    temp_cell = _operate_cell_temperature(temp_model, env_params)

    # TODO: Chunkify based on max memory usage set in TOML config file; progressively save to disk
    # TODO: Parallelize calculations across cores set in TOML config file

    timestamps = effective_irradiance.index
    num_timesteps = len(timestamps)

    iv_point_cols = ['i_sc', 'v_oc', 'i_mp', 'v_mp', 'p_mp', 'i_x', 'i_xx']
    index = MultiIndex.from_product([range(num_panels), timestamps], names=['panel_id', 'timestamp'])
    panels = DataFrame(0.0, index=index, columns=iv_point_cols)

    effect_names = get_effect_names()
    attribution = DataFrame(
        NaT,
        index=Index(range(num_panels), name='panel_id'),
        columns=[f"{name}_onset" for name in effect_names],
        dtype=timestamps.dtype,
    )

    for panel_num in range(num_panels):
        modifiers, onsets = compute_modifiers(panel_num)

        # Apply effective irradiance and temp_cell modifiers
        panel_irr = effective_irradiance * modifiers.get('G', 1.0)
        panel_temp = temp_cell + modifiers.get('T', 0.0)

        I_l, I_0, R_s, R_sh, nNsVth = pvsystem.calcparams_cec(
            effective_irradiance=panel_irr,
            temp_cell=panel_temp,
            alpha_sc=module_params['alpha_sc'],
            a_ref=module_params['a_ref'],
            I_L_ref=module_params['I_L_ref'],
            I_o_ref=module_params['I_o_ref'],
            R_sh_ref=module_params['R_sh_ref'],
            R_s=module_params['R_s'],
            Adjust=module_params['Adjust']
        )

        # Additive or multiplicative according to AGGREGATION_RULES (effect_generator)
        I_l  = I_l * modifiers.get('I_L',  1.0)
        I_0 = I_0 + modifiers.get('I_0', 0.0)
        R_s = R_s + modifiers.get('R_s', 0.0)
        R_sh = R_sh * modifiers.get('R_sh', 1.0)

        iv_points = pvsystem.singlediode(
            photocurrent=I_l,
            saturation_current=I_0,
            resistance_series=R_s,
            resistance_shunt=R_sh,
            nNsVth=nNsVth,
            method=method,
        )

        start = panel_num * num_timesteps
        panels.iloc[start:start + num_timesteps] = iv_points[iv_point_cols].values

        for effect_name, onset_step in onsets.items():
            attribution.at[panel_num, f"{effect_name}_onset"] = timestamps[onset_step]

    return panels, attribution
