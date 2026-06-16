# Moral Foundations Configuration
# Mapping trait indices to Moral Foundations based on Moral Foundations Theory (MFT)

FOUNDATIONS = {
    "Care": {
        84: 1,   # cruel :: kind
        301: -1, # empath :: psychopath
        179: 1   # poisonous :: nurturing
    },
    "Fairness": {
        14: 1,   # cunning :: honorable
        196: -1, # equitable :: hypocritical
        226: 1   # racist :: egalitarian
    },
    "Loyalty": {
        28: -1,  # loyal :: traitorous
        328: -1, # devoted :: unfaithful
        219: -1  # patriotic :: unpatriotic
    },
    "Authority": {
        134: -1, # obedient :: rebellious
        31: 1,   # rude :: respectful
        197: -1  # traditional :: unorthodox
    },
    "Sanctity": {
        81: -1,  # angelic :: demonic
        221: -1, # wholesome :: salacious
        64: 1    # debased :: pure
    },
    "Liberty": {
        163: -1, # independent :: codependent
        195: -1, # individualist :: communal
        134: 1   # obedient :: rebellious
    }
}

#Number of archetypes
NUM_COMPONENTS = 50
