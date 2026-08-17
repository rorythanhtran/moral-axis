# Moral Foundations Configuration (Agreed Traits from LLM Comparison)
# Mapping trait indices to Moral Foundations based on agreed traits in llm_comparison_agreement.tsv
# Polarities have been assigned where 1 represents the right pole as virtue/aligned and -1 represents the left pole as virtue/aligned.

FOUNDATIONS = {
    "Care": {
        22: 1,     # ferocious :: pacifist
        25: -1,    # forgiving :: vengeful
        76: 1,     # quarrelsome :: warm
        79: 1,     # selfish :: altruistic
        84: 1,     # cruel :: kind
        99: 1,     # remote :: involved
        128: -1,   # vulnerable :: armoured
        129: -1,   # warm :: cold
        179: 1,    # poisonous :: nurturing
        194: -1,   # complimentary :: insulting
        241: 1,    # traumatized :: flourishing
        251: -1,   # oppressed :: privileged
        253: -1,   # vegan :: cannibal
        254: -1,   # loveable :: punchable
        271: -1,   # narcissistic :: low self esteem
        280: 1,    # trolling :: triggered
        301: -1,   # empath :: psychopath
        315: 1,    # stingy :: generous
        329: 1,    # fearmongering :: reassuring
        333: 1,    # genocidal :: not genocidal
        338: 1,    # bad boy :: white knight
        342: -1,   # giving :: receiving
        353: -1,   # love-focused :: money-focused
        361: -1,   # touchy-feely :: distant
        367: -1,   # hugs :: handshakes
        373: -1,   # parental :: childlike
        385: -1,   # woke :: problematic
        387: -1,   # hippie :: militaristic
        395: -1,   # friendly :: unfriendly
        408: 1,    # harsh :: gentle
        409: 1,    # clinical :: heartfelt
        414: 1,    # catty :: supportive
    },
    "Fairness": {
        14: 1,     # cunning :: honorable
        25: -1,    # forgiving :: vengeful
        46: -1,    # straightforward :: cryptic
        63: -1,    # democratic :: authoritarian
        101: 1,    # biased :: impartial
        171: -1,   # open-minded :: close-minded
        196: -1,   # equitable :: hypocritical
        225: -1,   # feminist :: sexist
        226: 1,    # racist :: egalitarian
        251: -1,   # oppressed :: privileged
        285: -1,   # factual :: exaggerating
        319: 1,    # two-faced :: one-faced
        347: -1,   # frank :: sugarcoated
        354: -1,   # transparent :: machiavellian
        385: -1,   # woke :: problematic
    },
    "Loyalty": {
        28: -1,    # loyal :: traitorous
        146: -1,   # insider :: outsider
        156: 1,    # gossiping :: confidential
        164: -1,   # family-first :: work-first
        195: 1,    # individualist :: communal
        219: -1,   # patriotic :: unpatriotic
        328: -1,   # devoted :: unfaithful
    },
    "Authority": {
        8: -1,     # strict :: lenient
        14: 1,     # cunning :: honorable
        15: -1,    # orderly :: chaotic
        24: -1,    # dominant :: submissive
        28: -1,    # loyal :: traitorous
        31: 1,     # rude :: respectful
        42: 1,     # mischievous :: well behaved
        57: -1,    # bossy :: meek
        58: 1,     # barbaric :: civilized
        63: 1,     # democratic :: authoritarian
        115: 1,    # outlaw :: sheriff
        134: -1,   # obedient :: rebellious
        138: 1,    # proletariat :: bourgeoisie
        139: -1,   # alpha :: beta
        142: -1,   # charismatic :: uninspiring
        149: -1,   # sheeple :: conspiracist
        157: -1,   # official :: backdoor
        161: -1,   # captain :: first-mate
        166: 1,    # wild :: tame
        167: -1,   # prestigious :: disreputable
        168: 1,    # scandalous :: proper
        174: 1,    # apprentice :: master
        197: -1,   # traditional :: unorthodox
        201: 1,    # anarchist :: statist
        215: 1,    # liberal :: conservative
        251: 1,    # oppressed :: privileged
        267: 1,    # folksy :: presidential
        268: -1,   # corporate :: freelance
        275: 1,    # whippersnapper :: sage
        281: -1,   # tattle-tale :: f***-the-police
        305: -1,   # stuck-in-the-past :: forward-thinking
        322: -1,   # chivalrous :: businesslike
        324: -1,   # on-time :: tardy
        341: -1,   # yes-man :: contrarian
        351: 1,    # radical :: centrist
        373: -1,   # parental :: childlike
        387: 1,    # hippie :: militaristic
        392: -1,   # good-manners :: bad-manners
        434: -1,   # leader :: follower
        449: 1,    # maverick :: conformist
        450: -1,   # social climber :: nonconformist
        453: -1,   # sincere :: irreverent
        456: -1,   # stereotypical :: boundary breaking
        459: 1,    # likes change :: resists change
        461: -1,   # old-fashioned :: progressive
    },
    "Sanctity": {
        6: 1,      # lewd :: tasteful
        9: -1,     # refined :: rugged
        11: -1,    # innocent :: worldly
        20: -1,    # spiritual :: skeptical
        23: -1,    # modest :: flamboyant
        32: -1,    # diligent :: lazy
        33: 1,     # lustful :: chaste
        40: -1,    # attractive :: repulsive
        44: 1,     # indulgent :: sober
        45: 1,     # kinky :: vanilla
        50: -1,    # works hard :: plays hard
        58: 1,     # barbaric :: civilized
        64: 1,     # debased :: pure
        66: -1,    # frugal :: lavish
        75: 1,     # disorganized :: self-disciplined
        81: -1,    # angelic :: demonic
        83: -1,    # devout :: heathen
        108: 1,    # sickly :: healthy
        122: -1,   # human :: animalistic
        150: -1,   # neat :: messy
        165: 1,    # scruffy :: manicured
        168: 1,    # scandalous :: proper
        175: -1,   # straight :: queer
        182: 1,    # soulless :: soulful
        218: 1,    # hedonist :: monastic
        221: -1,   # wholesome :: salacious
        234: 1,    # stinky :: fresh
        236: 1,    # self-destructive :: self-improving
        253: -1,   # vegan :: cannibal
        290: 1,    # exhibitionist :: bashful
        300: -1,   # clean :: perverted
        312: 1,    # freak :: normie
        317: 1,    # extravagant :: thrifty
        359: -1,   # prudish :: flirtatious
        360: -1,   # innocent :: jaded
        362: 1,    # muddy :: washed
        370: 1,    # junkie :: straight edge
        388: 1,    # gluttonous :: moderate
        397: -1,   # delicate :: coarse
        410: 1,    # inappropriate :: seemly
        420: -1,   # divine :: earthly
        427: -1,   # blessed :: cursed
        431: 1,    # mechanical :: natural
        447: -1,   # love shy :: cassanova
        462: 1,    # gross :: hygienic
    },
    "Liberty": {
        48: -1,    # libertarian :: socialist
        57: -1,    # bossy :: meek
        63: -1,    # democratic :: authoritarian
        96: 1,     # politically correct :: edgy
        106: 1,    # enslaved :: emancipated
        120: 1,    # resigned :: resistant
        201: -1,   # anarchist :: statist
        215: -1,   # liberal :: conservative
        225: -1,   # feminist :: sexist
        251: -1,   # oppressed :: privileged
        268: 1,    # corporate :: freelance
        281: 1,    # tattle-tale :: f***-the-police
        351: -1,   # radical :: centrist
        372: 1,    # dystopian :: utopian
        387: -1,   # hippie :: militaristic
        437: -1,   # activist :: nonpartisan
        449: -1,   # maverick :: conformist
        450: 1,    # social climber :: nonconformist
    }
}

# Number of archetypes
NUM_COMPONENTS = 50
