The main dataset of interest is the largest one with N=2000 characters across 341 stories and measured on 464 traits.

Note: Traits are semantic differentials.

The datasets with 800 and 1600 characters are subsets, each one based on a release by openpsychometrics.org. These smaller datasets are stored in ./data/plain/archive/.

All traits, characters, and stories have been standardized across datasets.
- Some names are shortened, often to match the typical name of the character in a story.
- Spelling errors
- Mismatches in character/story names
- All emoji-emoji traits have been removed
- A repeat of one trait has been removed (soft-hard)
- Removals reduce 500 traits to 464 for the largest dataset

Files are in the same format for each dataset and are stored here:

./data/plain/0800/
./data/plain/1600/
./data/plain/2000/

Descriptions (referencing N=2000):

-----------------------------------------------------------

data_raw_A.txt
data_raw_Acounts.tsv
data_raw_Astds.txt

A: average ratings for each semantic differential and character pair.
- ratings are normalized to the range -0.5 to 0.5
- a negative rating aligns with the first bipolar adjective of the given
semantic differential
- a positive rating aligns with the first bipolar adjective of the given
semantic differential
Acounts: Number of ratings per pair
Astds: Standard deviation in ratings

-----------------------------------------------------------

data_traits_max_strength.txt

One number: Max magnitude of all trait vectors; used for normalization.

data_traits.tsv

Columns:
- index
- semantic differential (e.g., persistent--quitter)
- flipped differential
- left pole
- right pole
- card url

-----------------------------------------------------------

data_characters_max_strength.txt

One number: Max magnitude of all character vectors; used for normalization.


data_characters.tsv

Columns:
- index
- character
- character in LaTeX
- character/story
- character/story in LaTeX
- card url


-----------------------------------------------------------

data_stories.tsv

Columns:
- story titles
- story titles formatted for LaTeX
- story card urls

-----------------------------------------------------------

data_character_story_indices.txt

Ragged text file.

Each line has the index of the story then indices of characters in that story

e.g., Friends is the 99th story.
Line with indices for the characters is:
99 1198 1199 1200 1201 1202 1203

-----------------------------------------------------------

SVD matrices:

- data_archetype_space_U.txt
- data_archetype_space_Sigma.txt
- data_archetype_space_V.txt

232 archetype classes (order matters):

- data_archetype_space_archetype_classes.txt

Cosines, components, and variance explained for traits and characters:

- data_archetype_space_trait_alignment_cosines.txt
- data_archetype_space_trait_component_norms.txt
- data_archetype_space_trait_variance_explained.txt

- data_archetype_space_character_alignment_cosines.txt
- data_archetype_space_character_component_norms.txt
- data_archetype_space_character_variance_explained.txt

Archetype ratios:

data_archetype_space_character_archetype_ratios.txt

Major archetype: Ratio >= 10
Minor archetype: Ratio >= 5
Weak archetype: Ratio < 5

Archetype determined for each character; for dual and triple archetypes, ordering of archetypes is by variance explained for a character, increasing (linguistic ordering)

- data_archetype_space_character_archetypes_by_class.txt

Archetype class for each character where ordering is by overall singular values:

- data_archetype_space_character_archetypes_ordered.txt

Indices into data_archetype_space_archetype_classes.txt
for each character's archetype:

- data_archetype_space_character_archetype_indices.txt
