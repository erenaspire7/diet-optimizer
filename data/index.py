import pandas as pd


class Data:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Data, cls).__new__(cls)
            cls._instance.load()
        return cls._instance

    def load(self):
        self.df = pd.read_excel(
            "/home/erenaspire7/repos/evos/diet-optimizer/data/DietProblem.xls",
            sheet_name="Sheet1",
        )

        # cast to float
        numeric_columns = [col for col in self.df.columns if col != "Item per $1"]
        self.df[numeric_columns] = self.df[numeric_columns].astype("float64")

    def get_nutritional_value(self, item):
        return self.df.loc[self.df["Item per $1"] == item].copy()

    def get_items(self):
        return self.df["Item per $1"].tolist()

    def get_optimal_annual_intake(self):
        return pd.DataFrame(
            [
                {
                    "Kilo calories": 1095,
                    "protein (grams)": 25550,
                    "calcium (grams)": 292,
                    "iron (mg)": 4380,
                    "vitaminA 1000 IU": 1825000,
                    "thiamine (mg)": 657,
                    "riboflavin (mg)": 986,
                    "niacin (mg)": 6570,
                    "ascorbicAcid (mg)": 27375,
                }
            ],
        )

    def get_idealized_cost(self):
        return 40
