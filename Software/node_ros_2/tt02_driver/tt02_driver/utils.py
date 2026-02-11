"""
Some utility stuff
"""

# ======================================================
#  AINSI formats (colors, font styles, ...)
# ======================================================
def ARED(text: str) -> str:         # RED
    return f"\033[91m{text}\033[0m"
def AGREEN(text: str) -> str:       # GREEN
    return f"\033[92m{text}\033[0m"
def AYELLOW(text: str) -> str:      # YELLOW
    return f"\033[93m{text}\033[0m"
def ABLUE(text: str) -> str:        # BLUE
    return f"\033[94m{text}\033[0m"
def AMAGENTA(text: str) -> str:     # MAGENTA
    return f"\033[95m{text}\033[0m"
def ACYAN(text: str) -> str:        # CYAN
    return f"\033[96m{text}\033[0m" 
def ABOLD(text: str) -> str:        # BOLD
    return f"\033[1m{text}\033[0m"
def AUNDERLINE(text: str) -> str:   # UNDERLINE
    return f"\033[4m{text}\033[0m"




