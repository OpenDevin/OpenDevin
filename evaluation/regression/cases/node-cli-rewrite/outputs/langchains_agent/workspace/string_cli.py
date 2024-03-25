import sys

import commands


def print_help():
    help_text = """
Usage: python string_cli.py <command> <string>

Commands:
    reverse - Reverses the input string.
    uppercase - Converts the input string to uppercase.
    lowercase - Converts the input string to lowercase.
    spongebob - Converts the input string to spongebob case.
    length - Returns the length of the input string.
    scramble - Randomly scrambles the characters in the input string.
"""
    print(help_text)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--help":
        print_help()
        sys.exit(0)
    elif len(sys.argv) < 3:
        print("Usage: python string_cli.py <command> <string>")
        sys.exit(1)

    command = sys.argv[1]
    input_string = sys.argv[2]

    if command == "reverse":
        from commands.reverse import reverse_string

        print(reverse_string(input_string))
    elif command == "uppercase":
        from commands.uppercase import to_uppercase

        print(to_uppercase(input_string))
    elif command == "lowercase":
        from commands.lowercase import to_lowercase

        print(to_lowercase(input_string))
    elif command == "spongebob":
        from commands.spongebob import spongebob_case

        print(spongebob_case(input_string))
    elif command == "length":
        from commands.length import string_length

        print(string_length(input_string))
    elif command == "scramble":
        from commands.scramble import scramble_string

        print(scramble_string(input_string))
    else:
        print("Invalid command!")
