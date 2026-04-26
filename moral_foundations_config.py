# Moral Foundations Configuration
# Mapping trait indices to Moral Foundations based on Moral Foundations Theory (MFT) components 
# Direction: +1 for the Right pole, -1 for the Left pole. +1/-1 scales as it indicares perfect alignments/opposites. 

FOUNDATIONS = {
    "Care": {
        84: 1,   # cruel :: kind (Kind is Right)
        301: -1, # empath :: psychopath (Empath is Left)
        179: 1   # poisonous :: nurturing (Nurturing is Right)
    },
    "Fairness": {
        14: 1,   # cunning :: honorable (Honorable is Right)
        196: -1, # equitable :: hypocritical (Equitable is Left)
        226: 1   # racist :: egalitarian (Egalitarian is Right)
    },
    "Loyalty": {
        28: -1,  # loyal :: traitorous (Loyal is Left)
        328: -1, # devoted :: unfaithful (Devoted is Left)
        219: -1  # patriotic :: unpatriotic (Patriotic is Left)
    },
    "Authority": {
        134: -1, # obedient :: rebellious (Obedient is Left)
        31: 1,   # rude :: respectful (Respectful is Right)
        197: -1  # traditional :: unorthodox (Traditional is Left)
    },
    "Sanctity": {
        81: -1,  # angelic :: demonic (Angelic is Left)
        221: -1, # wholesome :: salacious (Wholesome is Left)
        64: 1    # debased :: pure (Pure is Right)
    },
    "Liberty": {
        163: -1, # independent :: codependent (Independent is Left)
        195: -1, # individualist :: communal (Individualist is Left)
        134: 1   # obedient :: rebellious (Rebellious is Right)
    }
}

# Number of archetype components to use for the analysis
NUM_COMPONENTS = 6
