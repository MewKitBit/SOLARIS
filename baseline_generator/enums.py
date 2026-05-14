from enum import Enum
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

class TemperatureModel(Enum):
    """Valid temperature models for PVLib"""
    SAPM_OPEN_RACK_GLASS = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    SAPM_CLOSE_MOUNT_GLASS = TEMPERATURE_MODEL_PARAMETERS['sapm']['close_mount_glass_glass']
    SAPM_OPEN_RACK_POLYMER = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
    SAPM_INSULATED_BACK_POLYMER = TEMPERATURE_MODEL_PARAMETERS['sapm']['insulated_back_glass_polymer']
    PVSYST_FREESTANDING = TEMPERATURE_MODEL_PARAMETERS['pvsyst']['freestanding']
    PVSYST_INSULATED = TEMPERATURE_MODEL_PARAMETERS['pvsyst']['insulated']
    PVSYST_SEMI_INTEGRATED = TEMPERATURE_MODEL_PARAMETERS['pvsyst']['semi_integrated']

class SingleDiodeMethod(Enum):
    """Valid single diode resolution methods for PVLib"""
    LAMBERTW = 'lambertw'
    NEWTON = 'newton'
    BRENTQ = 'brentq'
    CHANDRUPATLA = 'chandrupatla'

class ModuleSource(Enum):
    """Valid module sources for PVLib"""
    CEC = 'CECMod'

class IncidentAngleModel(Enum):
    """Valid incident angle models for PVLib"""
    ASHRAE = 'ashrae'
    PHYSICAL = 'physical'
    MARTIN_RUIZ = 'martin_ruiz'