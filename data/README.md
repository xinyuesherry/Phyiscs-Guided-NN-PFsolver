# Simulate the power system operation measurement data

- Real-world load data

  The parsed datasets are uploaded. To refer to raw data, see the links below:
  - [Kaggle load data](https://www.kaggle.com/c/global-energy-forecasting-competition-2012-load-forecasting/overview/description)
  - [PGE load data](https://www.pge.com/en_US/residential/save-energy-money/analyze-your-usage/energy-data-hub/energy-data-hub-for-customers-and-third-parties.page)

- Power system simulator
   - [MATPOWER](https://github.com/MATPOWER/matpower)
  
- Code to generate simulation data
  - [a reference](https://github.com/YuxiaoLiu/data-driven-power-flow-linearization/blob/master/DataGeneration.m) from repo [data-driven-power-flow-linearization](https://github.com/YuxiaoLiu/data-driven-power-flow-linearization/tree/master)
  - ***Usage***: Do not use random load mode. Instead, import the above real-world load data, subsample or scale to match your case.
    
