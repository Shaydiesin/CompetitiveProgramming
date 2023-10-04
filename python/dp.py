
str = "123.737"
parts = []
current_part = ""


for char in str:

    if char.isalnum():
        current_part += char
    else:
        if current_part:
            parts.append(current_part)
        current_part = ""


if current_part:
    parts.append(current_part)

print(len(parts))