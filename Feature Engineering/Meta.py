import pandas as pd
from mafese import Data
from mafese.wrapper.mha import MhaSelector
from mafese import evaluator
from sklearn.svm import SVC
from mafese.utils import transfer

# Load the dataset
dataset = pd.read_csv('0.9_5subjectslabelled_data.csv', index_col=0).values
X, y = dataset[:, 0:-1], dataset[:, -1]
data = Data(X, y)

data.split_train_test(test_size=0.2, inplace=True)

# Define the list of possible values for problem, estimator, and optimizer
problems = ["classification", "regression"]
estimators = ["knn"]
optimizers = ["L_SHADE", "SADE", "SAP_DE", "SHADE", "DevDMOA", "OriginalDMOA", "OriginalDO", "BaseEFO", "OriginalEFO", "OriginalEHO",
              "AdaptiveEO", "ModifiedEO", "OriginalEO", "OriginalEOA", "LevyEP", "OriginalEP", "CMA_ES",
              "LevyES", "OriginalES", "Simple_CMA_ES", "OriginalESOA", "OriginalEVO", "OriginalFA", "BaseFBIO",
              "OriginalFBIO", "OriginalFFA", "OriginalFFO", "OriginalFLA", "BaseFOA", "OriginalFOA", "WhaleFOA",
              "OriginalFOX", "OriginalFPA", "BaseGA", "EliteMultiGA", "EliteSingleGA", "MultiGA", "SingleGA",
              "OriginalGBO", "BaseGCO", "OriginalGCO", "OriginalGJO", "OriginalGOA", "BaseGSKA", "OriginalGSKA",
              "Matlab101GTO", "Matlab102GTO", "OriginalGTO", "GWO_WOA", "OriginalGWO", "RW_GWO", "OriginalHBA",
              "OriginalHBO", "OriginalHC", "SwarmHC", "OriginalHCO", "OriginalHGS", "OriginalHGSO", "OriginalHHO",
              "BaseHS", "OriginalHS", "OriginalICA", "OriginalINFO", "OriginalIWO", "BaseJA", "LevyJA",
              "OriginalJA", "BaseLCO", "ImprovedLCO", "OriginalLCO", "OriginalMA", "BaseMFO", "OriginalMFO",
              "OriginalMGO", "OriginalMPA", "OriginalMRFO", "WMQIMRFO", "OriginalMSA", "BaseMVO", "OriginalMVO",
              "OriginalNGO", "ImprovedNMRA", "OriginalNMRA", "OriginalNRO", "OriginalOOA", "OriginalPFA",
              "OriginalPOA", "CL_PSO", "C_PSO", "HPSO_TVAC", "OriginalPSO", "PPSO", "OriginalPSS", "BaseQSA",
              "ImprovedQSA", "LevyQSA", "OppoQSA", "OriginalQSA", "OriginalRIME", "OriginalRUN", "OriginalSA",
              "BaseSARO", "OriginalSARO", "BaseSBO", "OriginalSBO", "BaseSCA", "OriginalSCA", "QleSCA",
              "OriginalSCSO", "ImprovedSFO", "OriginalSFO", "OriginalSHIO", "OriginalSHO", "ImprovedSLO",
              "ModifiedSLO", "OriginalSLO", "BaseSMA", "OriginalSMA", "DevSOA", "OriginalSOA", "OriginalSOS",
              "DevSPBO", "OriginalSPBO", "OriginalSRSR", "BaseSSA", "OriginalSSA", "OriginalSSDO", "OriginalSSO",
              "OriginalSSpiderA", "OriginalSSpiderO", "OriginalSTO", "OriginalSeaHO", "OriginalServalOA",
              "OriginalTDO", "BaseTLO", "ImprovedTLO", "OriginalTLO", "OriginalTOA", "OriginalTPO", "OriginalTSA",
              "OriginalTSO", "EnhancedTWO", "LevyTWO", "OppoTWO", "OriginalTWO", "BaseVCS", "OriginalVCS",
              "OriginalWCA", "OriginalWDO", "OriginalWHO", "HI_WOA", "OriginalWOA", "OriginalWaOA",
              "OriginalWarSO", "OriginalZOA"]

# Iterate over all combinations
for problem in problems:
    for estimator in estimators:
        for optimizer in optimizers:
            # Define the feature selection method
            feat_selector = MhaSelector(problem=problem, estimator=estimator,
                                        optimizer=optimizer, optimizer_paras=None,
                                        transfer_func="vstf_01", obj_name="AS")

            # Fit the feature selector to the training data
            feat_selector.fit(data.X_train, data.y_train, fit_weights=(0.9, 0.1), verbose=True)

            # Perform feature selection and evaluation
            X_train_selected = feat_selector.transform(data.X_train)
            X_test_selected = feat_selector.transform(data.X_test)

            results = evaluator.evaluate(feat_selector, estimator=SVC(), data=data, metrics=["AS"])
            print(f"Results for problem={problem}, estimator={estimator}, optimizer={optimizer}:")
            print(results)
            print("--------------------------------------------------")
