import pvlib.irradiance

from enums import TemperatureModel, SingleDiodeMethod, IncidentAngleModel
from pandas import DataFrame
from pvlib import pvsystem, temperature, iam

def __operate_effective_irradiance(module_params, env_params, solar_positions, iam_model):
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

    aoi = pvlib.irradiance.aoi(
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

def __operate_cell_temperature(temp_model: TemperatureModel, env_params):
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

def __operate_cec(module_params: dict, env_params: DataFrame, solar_positions: DataFrame, method: SingleDiodeMethod,
                  temp_model: TemperatureModel, iam_model:  IncidentAngleModel) -> DataFrame:
    """
    Executes the complete California Energy Commission (CEC) single-diode pipeline to generate
    the operational I-V curve parameters for a specific PV module over time.

    :param module_params: Dictionary of the physical and electrical parameters of the module.
                          Must include mounting geometry, IAM glass parameters, and the CEC
                          reference parameters (``alpha_sc``, ``a_ref``, ``I_L_ref``, ``I_o_ref``,
                          ``R_sh_ref``, ``R_s``, ``Adjust``).
    :param env_params: ``pd.DataFrame`` containing weather and Plane of Array (POA) irradiances.
    :param solar_positions: ``pd.DataFrame`` containing the time-series geometric solar positions.
    :param method: ``SingleDiodeMethod`` enum specifying the mathematical solver for the single-diode equation.
    :param temp_model: ``TemperatureModel`` enum for the cell temperature estimation.
    :param iam_model: ``IncidentAngleModel`` enum for the effective irradiance estimation.
    :return: ``pd.DataFrame`` containing the calculated five parameters (I_L, I_0, R_s, R_sh, nNsVth)
             and the resulting primary I-V points (i_sc, v_oc, i_mp, v_mp, p_mp, i_x, i_xx) as a time-series.
    """

    effective_irradiance = __operate_effective_irradiance(module_params, env_params, solar_positions, iam_model)
    temp_cell = __operate_cell_temperature(temp_model, env_params)

    I_l, I_0, R_s, R_sh, nNsVth = pvsystem.calcparams_cec(
        effective_irradiance=effective_irradiance,
        temp_cell=temp_cell,
        alpha_sc=module_params['alpha_sc'],
        a_ref=module_params['a_ref'],
        I_L_ref=module_params['I_L_ref'],
        I_o_ref=module_params['I_o_ref'],
        R_sh_ref=module_params['R_sh_ref'],
        R_s=module_params['R_s'],
        Adjust=module_params['Adjust']
    )

    i_v_curve = pvsystem.singlediode(
        photocurrent=I_l,
        saturation_current=I_0,
        resistance_series=R_s,
        resistance_shunt=R_sh,
        nNsVth=nNsVth,
        method=method
    )

    return i_v_curve