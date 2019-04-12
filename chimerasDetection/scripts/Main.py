'''
This script only launches the necessary top-level functions.
Read detailed descriptions and change inside arguments of these functions in set_arguments.py
'''

from set_arguments import *

def main():
    get_data()
    for_parsing()
    create_db()
    for_nn1()

main()