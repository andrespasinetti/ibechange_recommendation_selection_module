# content_selection/mab_initialiser.py

import logging
import json
import os

from cs_module.multi_armed_bandit.logistic_laplace_ts import LogisticLaplaceTS
from cs_module.multi_armed_bandit.bernoulli_beta_ts import BernoulliBetaTS
from cs_module.multi_armed_bandit.random_bandit import RandomBandit
from cs_module.multi_armed_bandit.recommendation_optimal_bandit import RecommendationOptimalBandit
from cs_module.multi_armed_bandit.resource_optimal_bandit import ResourceOptimalBandit

from cs_module.utils.encoding import get_dim_intervention_feature_vector

from cs_module.config import INTERVENTION_MAB_CONFIG, RECOMMENDATION_MAB_CONFIG, RESOURCE_MAB_CONFIG

logger = logging.getLogger(__name__)


class MABInitialiser:
    def __init__(self, data_storage):
        self.data_storage = data_storage

        self.resource_mab = self._init_resource_mab()
        self.intervention_mab = self._init_intervention_mab()
        self.recommendation_mab = self._init_recommendation_mab()

    def _init_resource_mab(self):
        t = RESOURCE_MAB_CONFIG["type"]
        logger.info(f"Initializing resource MAB with mode: {t}...")

        if t == "BernoulliBetaTS":
            kwargs = {}
            if "alpha_0" in RESOURCE_MAB_CONFIG:
                kwargs["alpha_0"] = RESOURCE_MAB_CONFIG["alpha_0"]
            if "beta_0" in RESOURCE_MAB_CONFIG:
                kwargs["beta_0"] = RESOURCE_MAB_CONFIG["beta_0"]
            mab = BernoulliBetaTS(**kwargs)

        elif t == "ResourceOptimalBandit":
            path = os.path.join(os.path.dirname(__file__), "..", "user_preferences", "res_preferences.json")
            with open(path) as f:
                resource_pref = json.load(f)
            mab = ResourceOptimalBandit(resource_pref=resource_pref)

        elif t == "RandomBandit":
            mab = RandomBandit()

        else:
            raise ValueError(f"Unknown resource_mab_mode: {t}")

        self.data_storage.initialize_bandit(
            table="resource_mab_runs", bandit_type=t, initial_params=mab.initial_parameters
        )
        return mab

    def _init_intervention_mab(self):
        t = INTERVENTION_MAB_CONFIG["type"]
        logger.info(f"Initializing intervention MAB with mode: {t}...")

        if t == "LogisticLaplaceTS":
            dim = get_dim_intervention_feature_vector(include_bias=True)
            mab = LogisticLaplaceTS(feature_dim=dim)

        elif t == "None":
            return None

        else:
            raise ValueError(f"Unknown intervention_mab_mode: {t}")

        self.data_storage.initialize_bandit(
            table="intervention_mab_runs", bandit_type=t, initial_params=mab.initial_parameters
        )
        return mab

    def _init_recommendation_mab(self):
        t = RECOMMENDATION_MAB_CONFIG["type"]
        logger.info(f"Initializing recommendation MAB with mode: {t}...")

        if t == "BernoulliBetaTS":
            kwargs = {}
            if "alpha_0" in RECOMMENDATION_MAB_CONFIG:
                kwargs["alpha_0"] = RECOMMENDATION_MAB_CONFIG["alpha_0"]
            if "beta_0" in RECOMMENDATION_MAB_CONFIG:
                kwargs["beta_0"] = RECOMMENDATION_MAB_CONFIG["beta_0"]
            mab = BernoulliBetaTS(**kwargs)

        elif t == "RecommendationOptimalBandit":
            int_path = os.path.join(os.path.dirname(__file__), "..", "user_preferences", "int_preferences.json")
            with open(int_path) as f:
                intervention_pref = json.load(f)

            rec_path = os.path.join(os.path.dirname(__file__), "..", "user_preferences", "rec_preferences.json")
            with open(rec_path) as f:
                recommendation_pref = json.load(f)

            mab = RecommendationOptimalBandit(
                intervention_pref=intervention_pref, recommendation_pref=recommendation_pref
            )

        elif t == "RandomBandit":
            mab = RandomBandit()

        else:
            raise ValueError(f"Unknown recommendation_mab_mode: {t}")

        self.data_storage.initialize_bandit(
            table="recommendation_mab_runs", bandit_type=t, initial_params=mab.initial_parameters
        )
        return mab
