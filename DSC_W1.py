def example_word_count():
    # This example question requires counting words in the example_string below.
    example_string = "Amy is 5 years old"

    # YOUR CODE HERE.
    # You should write your solution here, and return your result, you can comment out or delete the
    # NotImplementedError below.
    result = example_string.split(" ")
    return len(result)

    #raise NotImplementedError()

import re
def names():
    simple_string = """Amy is 5 years old, and her sister Mary is 2 years old.
    Ruth and Peter, their parents, have 3 kids."""
    return(re.findall('[A-Z][a-z]*',simple_string))

    # YOUR CODE HERE
    #raise NotImplementedError()


import re
def grades():
    with open ("assets/grades.txt", "r") as file:
        grades = file.read()
        AA=re.findall('(?P<name>[\w ]*): B',grades)
        return (AA)
grades()

    # YOUR CODE HERE
   #raise NotImplementedError()




import re
def logs():
    with open("assets/logdata.txt", "r") as file:
        logdata = file.read()
        B=[]
        for A in re.finditer('(?P<host>[0-9.]{8,18})(\s-\s)(?P<user_name>[[a-z]{1,20}[0-9]{4}|-)(\s\[)(?P<time>[0-9]{2}/[A-Z][a-z]{2}/[0-9:]{13}\s-[0-9]{4})(]\s")(?P<request>[A-Z]*\s/[\w /+%-]*/[0-9.]{3})',logdata):
            B.append(A.groupdict())
        return B
