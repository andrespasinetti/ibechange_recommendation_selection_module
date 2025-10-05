import random
from virtual_user.config import PILLARS
from virtual_user.utils.get_pillar import get_pillar


class UserMissionManager:
    def __init__(self, user, missions):
        self.user = user
        self.missions = missions  # All missions available

    def select_new_missions(self, mode="random"):
        user_new_missions = []
        if self.user.current_missions:  # If the user has missions from the previous week
            for previous_mission_id in self.user.current_missions:
                if mode == "random":  # Select a random pillar then a random mission
                    new_pillar = random.choice(PILLARS)
                    possible_missions = [
                        mission_id for mission_id in self.missions if get_pillar(mission_id) == new_pillar
                    ]
                    user_new_missions.append(random.choice(possible_missions))

                elif mode == "fixed":
                    # Keep previous pillar and mission
                    user_new_missions.append(previous_mission_id)

                elif mode == "user_keep_pillar":
                    if random.random() < self.user.profile["mission_retain_probability"]:
                        user_new_missions.append(previous_mission_id)  # Retain the mission
                    else:  # change mission
                        new_missions = [
                            mission_id
                            for mission_id in self.missions
                            if get_pillar(mission_id) == get_pillar(previous_mission_id)
                            and mission_id != previous_mission_id
                        ]
                        if new_missions:
                            user_new_missions.append(
                                random.choice(new_missions)
                            )  # Choose a new mission from the same pillar
                        else:
                            user_new_missions.append(None)  # No missions left in the pillar

                elif mode == "user_specific":
                    if random.random() < self.user.profile["pillar_retain_probability"]:  # retain pillar
                        if random.random() < self.user.profile["mission_retain_probability"]:
                            user_new_missions.append(previous_mission_id)  # Retain the mission
                        else:
                            new_missions = [
                                mission_id
                                for mission_id in self.missions
                                if get_pillar(mission_id) == get_pillar(previous_mission_id)
                                and mission_id != previous_mission_id
                            ]
                            if new_missions:
                                user_new_missions.append(
                                    random.choice(new_missions)
                                )  # Choose a new mission from the same pillar
                            else:
                                user_new_missions.append(None)  # No missions left in the pillar
                    else:  # change pillar uniformly
                        new_pillar = random.choice([p for p in PILLARS if p != get_pillar(previous_mission_id)])
                        new_missions = [
                            mission_id for mission_id in self.missions if get_pillar(mission_id) == new_pillar
                        ]
                        if new_missions:
                            user_new_missions.append(
                                random.choice(new_missions)
                            )  # Choose a new mission from the same pillar
                        else:
                            user_new_missions.append(None)  # No missions left in the pillar

        else:
            user_new_missions.append(
                random.choice(list(self.missions.keys()))
            )  # No previous missions, choose one randomly

        return user_new_missions
