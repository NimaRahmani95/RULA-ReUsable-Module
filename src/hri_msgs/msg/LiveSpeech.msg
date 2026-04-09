# This message encodes the live result of a speech recognition process.
# A series of incremental results might be provided, until a final recognition
# hypothesis is returned.

std_msgs/Header header

string incremental # incremental recognition results
string final # final recognised text

float64 confidence # confidence of the final recognised speech, [0-1]

# The desired locale, using following format:
# the [ISO 639-1 language code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes), 
# lower case, followed by an underscore, 
# followed by the [ISO 3166-1 alpha-2 region code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2), 
# upper case.
#
# For example, 'en_GB' specifies British English.
string locale

#### TMP -- to be removed
string language
