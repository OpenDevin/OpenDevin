def spongebob_case(s):
    result = ""
    for i, char in enumerate(s):
        if i % 2 == 0:
            result += char.lower()
        else:
            result += char.upper()
    return result
