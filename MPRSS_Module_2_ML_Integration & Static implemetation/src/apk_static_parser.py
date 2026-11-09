"""
MPRSS APK Static Parser - Module 2 (real APK input)
Opens a real .apk file with androguard and extracts the same 215 feature
names used to train permission_risk_rf.pkl, so it can be fed straight into
the Flask API (api/app.py) as 'active_features'.
"""

from loguru import logger
import sys
from androguard.misc import AnalyzeAPK
import json
from loguru import logger
import sys
logger.remove()
logger.add(sys.stderr, level="WARNING")

import logging
logging.getLogger('androguard').setLevel(logging.WARNING)


class APKStaticParser:
    """Parse a real .apk into a Drebin-215-compatible active_features list"""

    def __init__(self, feature_metadata_path='models/feature_columns.json'):
        with open(feature_metadata_path) as f:
            metadata = json.load(f)

        self.feature_columns    = metadata['feature_columns']
        self.feature_categories = metadata['feature_categories']

        self.permission_features = [f for f, c in self.feature_categories.items()
                                     if c == 'Manifest Permission']
        self.intent_features     = [f for f, c in self.feature_categories.items()
                                     if c == 'Intent']
        self.api_call_features   = [f for f, c in self.feature_categories.items()
                                     if c == 'API call signature']
        self.command_features    = [f for f, c in self.feature_categories.items()
                                     if c == 'Commands signature']

    # ------------------------------------------------------------------
    def parse(self, apk_path, verbose=True):
        """
        Analyze a real .apk file and return a dict:
            {
                'active_features': [...],           # ready for the Flask API
                'breakdown': {category: [...] },    # for transparency / debugging
                'app_info': {package, permissions_declared, ...}
            }
        """
        if verbose:
            print(f"\n{'='*50}")
            print(f"Parsing APK: {apk_path}")
            print(f"{'='*50}")

        a, d, dx = AnalyzeAPK(apk_path)
        dex_list = d if isinstance(d, list) else [d]

        active_features = set()
        breakdown = {'Manifest Permission': [], 'Intent': [],
                     'API call signature': [], 'Commands signature': []}

        # --- 1. Manifest Permissions ---
        raw_perms = a.get_permissions()  # e.g. 'android.permission.SEND_SMS'
        perm_short_names = {p.split('.')[-1] for p in raw_perms}
        for feat in self.permission_features:
            if feat in perm_short_names:
                active_features.add(feat)
                breakdown['Manifest Permission'].append(feat)

        # --- 2. Intents (actions declared in manifest / intent-filters) ---
        try:
            declared_actions = set(a.get_elements("action", "name"))
        except Exception:
            declared_actions = set()
        for feat in self.intent_features:
            if feat in declared_actions:
                active_features.add(feat)
                breakdown['Intent'].append(feat)

        # --- 3. Command signatures (shell command strings hardcoded in code) ---
        all_strings = set()
        for dex in dex_list:
            all_strings.update(dex.get_strings())
        for feat in self.command_features:
            if any(feat in s for s in all_strings):
                active_features.add(feat)
                breakdown['Commands signature'].append(feat)

        # --- 4. API call signatures (method/class names referenced in the code) ---
        method_tokens = set()
        for m in dx.get_methods():
            enc = m.get_method()
            method_tokens.add(enc.get_name())                                   
            cls = enc.get_class_name().replace('/', '.').strip('L;')            
            method_tokens.add(cls)
            method_tokens.add(f"{cls}.{enc.get_name()}")                      

        for feat in self.api_call_features:
            if feat in method_tokens:
                active_features.add(feat)
                breakdown['API call signature'].append(feat)

        result = {
            'active_features': sorted(active_features),
            'breakdown': breakdown,
            'app_info': {
                'package_name':          a.get_package(),
                'app_name':              a.get_app_name(),
                'permissions_declared':  len(raw_perms),
                'total_active_features': len(active_features),
            }
        }

        if verbose:
            print(f"Package:              {result['app_info']['package_name']}")
            print(f"Permissions declared: {result['app_info']['permissions_declared']}")
            print(f"Matched features:     {result['app_info']['total_active_features']} / {len(self.feature_columns)}")
            for cat, feats in breakdown.items():
                print(f"  {cat}: {len(feats)}")

        return result


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    apk_path = sys.argv[1] if len(sys.argv) > 1 else 'test_apks/sample.apk'

    parser = APKStaticParser('models/feature_columns.json')
    result = parser.parse(apk_path)

    print(f"\nActive features ready for API:")
    print(json.dumps(result['active_features'], indent=2))