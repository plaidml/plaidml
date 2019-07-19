"""Manages loading and saving settings into the environment."""

import json
import os
import sys
import uuid

import plaidml.exceptions


def _find_config(name):
    prefixes = [
        sys.prefix,
        os.path.join(sys.prefix, 'local'),
    ]
    for prefix in prefixes:
        cfg_path = os.path.join(prefix, 'share', 'plaidml', name)
        if os.path.exists(cfg_path):
            return cfg_path
    return None


def _setup_config(env_var, filename):
    if env_var not in os.environ:
        cfg_path = _find_config(filename)
        if cfg_path:
            os.environ[env_var] = cfg_path
        elif 'RUNFILES_DIR' not in os.environ:
            raise plaidml.exceptions.PlaidMLError(
                'Could not find PlaidML configuration file: "{}".'.format(filename))


_setup_config('PLAIDML_EXPERIMENTAL_CONFIG', 'experimental.json')
_setup_config('PLAIDML_DEFAULT_CONFIG', 'config.json')

CONFIG = 'PLAIDML_CONFIG'
CONFIG_FILE = 'PLAIDML_CONFIG_FILE'
DEVICE_IDS = 'PLAIDML_DEVICE_IDS'
EXPERIMENTAL = 'PLAIDML_EXPERIMENTAL'
SESSION = 'PLAIDML_SESSION'
SETTINGS = 'PLAIDML_SETTINGS'
TELEMETRY = 'PLAIDML_TELEMETRY'
ENABLE_WINOGRAD = 'PLAIDML_ENABLE_WINOGRAD'
USE_STRIPE = 'USE_STRIPE'
DEFAULT_CONFIG = 'PLAIDML_DEFAULT_CONFIG'
EXPERIMENTAL_CONFIG = 'PLAIDML_EXPERIMENTAL_CONFIG'

ENV_SETTINGS = [
    CONFIG,
    CONFIG_FILE,
    DEVICE_IDS,
    EXPERIMENTAL,
    SESSION,
    TELEMETRY,
    ENABLE_WINOGRAD,
]

USER_SETTINGS = os.path.expanduser(os.path.join('~', '.plaidml'))
SYSTEM_SETTINGS = os.path.normpath('/etc/plaidml')


#TODO(T1192): Push into plaidml.cc once stabilized
class _Settings(object):
    """Manages loading settings into the current environment.
    Manual settings take precedence over the environment.
    Environment settings take precedence over those in files.
    """

    def __init__(self):
        self._load()
        self._setup = False

    def start_session(self):
        """If there are any settings, start the session, otherwise fail."""
        if self.session:
            return
        elif not self.setup:
            raise plaidml.exceptions.PlaidMLError('PlaidML is not configured. Run plaidml-setup.')
        self.session = str(uuid.uuid4())  # Random session id

    def _setup_for_test(self, user_settings='', system_settings=''):
        """Sets environment for tests."""
        self._setup = False
        self.session = None
        global USER_SETTINGS, SYSTEM_SETTINGS
        USER_SETTINGS = user_settings
        SYSTEM_SETTINGS = system_settings
        for k in ENV_SETTINGS + [SETTINGS]:
            if k in os.environ:
                del os.environ[k]

    def _load(self):
        settings = {}
        settings_file = os.environ.get(SETTINGS, '')
        if not os.path.exists(settings_file):
            settings_file = USER_SETTINGS
        if not os.path.exists(settings_file):
            settings_file = SYSTEM_SETTINGS
        if os.path.exists(settings_file):
            for k, val in json.load(open(settings_file)).items():
                if k not in ENV_SETTINGS:
                    raise plaidml.exceptions.OutOfRange('Invalid key "{0}" in config {1}'.format(
                        k, settings_file))
                settings[k] = val
        for k, val in settings.items():
            if k not in os.environ:
                setattr(self, k.replace("PLAIDML_", "").lower(), val)

    def save(self, filename):
        settings = {}
        for k in ENV_SETTINGS:
            if k in os.environ and k is not "PLAIDML_SESSION":
                settings[k] = getattr(self, k.replace("PLAIDML_", "").lower())
        with open(filename, "w") as out:
            json.dump(settings, out, sort_keys=True, indent=4, separators=(',', ':'))

    @property
    def setup(self):
        settings_count = sum([1 if k in os.environ else 0 for k in ENV_SETTINGS])
        return settings_count != 0 or self._setup

    @setup.setter
    def setup(self, val):
        self._setup = val

    @property
    def user_settings(self):
        return USER_SETTINGS

    @property
    def system_settings(self):
        return SYSTEM_SETTINGS

    @property
    def config_file(self):
        return os.environ.get(CONFIG_FILE, None)

    @config_file.setter
    def config_file(self, val):
        os.environ[CONFIG_FILE] = val

    @property
    def config(self):
        return os.environ.get(CONFIG, None)

    @config.setter
    def config(self, val):
        if val is None:
            if CONFIG in os.environ:
                del os.environ[CONFIG]
        else:
            os.environ[CONFIG] = val

    @property
    def device_ids(self):
        ids = os.environ.get(DEVICE_IDS)
        return ids.split(' ') if ids else []

    @device_ids.setter
    def device_ids(self, val):
        os.environ[DEVICE_IDS] = ' '.join(val)

    @property
    def experimental(self):
        return os.environ.get(EXPERIMENTAL, '0') is not '0'

    @experimental.setter
    def experimental(self, val):
        os.environ[EXPERIMENTAL] = '1' if val else '0'

    @property
    def session(self):
        """Returns a session id if a session is running."""
        return os.environ.get(SESSION, None)

    @session.setter
    def session(self, val):
        if val is None:
            if SESSION in os.environ:
                del os.environ[SESSION]
        else:
            os.environ[SESSION] = val

    @property
    def telemetry(self):
        return os.environ.get(TELEMETRY, '0') is not '0'

    @telemetry.setter
    def telemetry(self, val):
        os.environ[TELEMETRY] = '1' if val else '0'

    @property
    def enable_winograd(self):
        return os.environ.get(ENABLE_WINOGRAD, '0') is not '0'
        #or os.environ.get( USE_STRIPE, '0') is not '0'

    @enable_winograd.setter
    def enable_winograd(self, val):
        os.environ[ENABLE_WINOGRAD] = '1' if val else '0'

    @property
    def default_config(self):
        return os.environ.get(DEFAULT_CONFIG)

    @property
    def experimental_config(self):
        return os.environ.get(EXPERIMENTAL_CONFIG)


module = _Settings()
module.__name__ = __name__
module._module = sys.modules[module.__name__]
module._pmodule = module
sys.modules[module.__name__] = module
