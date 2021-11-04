'''Alignment algorithm implemetatiom'''

from PIL import Image
import numpy as np
from Bio.pairwise2 import format_alignment
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.Alphabet import generic_rna
import sys
import glob
from PIL import Image


def img2dot(img):
    def binarize_output(img, coeff=0.25):
        size = len(img)
        for i in range(size):
            for j in range(size):
                if i != j:
                    if img[i][j] > 255 * coeff:
                        img[i][j] = 255
                    else:
                        img[i][j] = 0
                else:
                    img[i][j] = max(32, min(128, round(float(img[i][j]) / 32) * 32))
        return img

    def get_dot(img):
        codes = {32: 'A', 64: 'C', 96: 'G', 128: 'U'}
        size = len(img)
        coords = []
        seq = ''
        for i in range(size):
            for j in range(size):
                if i == j and img[i][j] != 0:
                    seq += codes[img[i][j]]
                if img[i][j] == 255 and i != j:
                    coords.append((min(i, j), max(i, j)))
        coords = set(coords)
        dot = ['.' for i in range(size)]
        for coord in coords:
            if dot[coord[0]] != "." or dot[coord[1]] != ".":
                continue
            if dot[coord[0]] == '.':
                dot[coord[0]] = '('
            if dot[coord[1]] == '.':
                dot[coord[1]] = ')'
        return seq, ''.join(dot)
    return get_dot(binarize_output(img))


def align_sequence(seq, dot, increase_perc=0.5, min_pairs=3,connectivity_threshold=5): # Min pairs - number of pairs in alignment to accout for it
    result = ["." for _ in range(len(dot))]
    edges = [] # Contains pairs of left and right indices
    stack = []

    def process_stem(ix):
        c = dot[ix]
        while c != ")" and ix != len(dot):  # Search for group of brackets
            if c == "(":
                stack.append(ix)
            ix += 1
            if ix == len(dot):
                return(ix)
            c = dot[ix]
        left_left = None # Borders for 2 points of condensation
        left_right = None # They are inclusive, so we should be always allowed to write to those positions
        right_left = None
        right_right = None
        if len(stack) == 0 or ix == len(dot):
            return -1
        while c != "(" and ix != len(dot):  # End of group
            if c == ")":
                if right_left is None:
                    right_left = ix
                if left_left is None or left_left - stack[-1] <= connectivity_threshold:
                    right_right = ix
                    t = stack.pop()
                    if left_right is None:
                        left_right = t
                    left_left = t # Ищем первый индекс
                else:
                    break
            ix += 1
            if ix == len(dot):
                if len(stack) == 0:
                    break
                else:
                    print("Brackets don't match!")
                    exit(0)
            c = dot[ix]

        perc = increase_perc
        width_left = left_right - left_left
        width_right = right_right - right_left
        # We have 2 points of condensation - for left border and for right
        # Equal step in both directions
        left_left = 0 if left_left - perc * width_left < 0 else int(left_left - perc * width_left)
        right_right = len(dot) - 1 if right_right + perc * width_right \
            >= len(dot) else int(right_right + (perc * width_right))
        left_right = int(left_right + perc * width_left)
        right_left = int(right_left - (perc * width_right))
        if left_right >= right_left:
            dif = int((left_right - right_left)/2)
            right_left += dif
            left_right = right_left - 1

        for l_edge,r_edge in edges: # Fix intervals if they intersect
            # Watch out for strict signs. Maybe it needs to be changed
            if left_left < l_edge and left_right >= l_edge and left_right <= r_edge:
                left_right = l_edge - 1
            if left_left>=l_edge and left_right <=r_edge:
                return ix # Inside another block - skip this one
            if left_right > r_edge and left_left >= l_edge and left_left <= r_edge:
                left_left = r_edge + 1
            if left_left < l_edge and left_right > r_edge: # Another block is inside
                if l_edge - left_left > left_right-r_edge:
                    left_right = l_edge - 1
                else:
                    left_left = r_edge + 1
            if right_left < l_edge and right_right >= l_edge and right_right <= r_edge:
                right_right = l_edge - 1
            if right_left>=l_edge and right_right <=r_edge:
                return ix # Inside another block - skip this one
            if right_right > r_edge and right_left >= l_edge and right_left <= r_edge:
                right_left = r_edge + 1
            if right_left < l_edge and right_right > r_edge: # Another block is inside
                if l_edge - right_left > right_right-r_edge:
                    right_right = l_edge - 1
                else:
                    right_left = r_edge + 1

        left = seq[left_left:left_right + 1]
        right = seq[right_left:right_right + 1]
        if len(left) == 0 or len(right) == 0:
            return ix
        # We will search for stem here
        my_rna = Seq(right, generic_rna)
        # We align complement here, so we will need to return to original sequence
        right = my_rna.reverse_complement()
        al = pairwise2.align.localms(left, right, 100, -150, -100,
                                     -100, one_alignment_only = True)  # Match, mismatch, open, extend
        if len(al) == 0:
            return ix
        # TODO: Choose best variant. Now we take the first one
        al_left_full, al_right_full, _, start, end = al[0]
        # print(al_left_full, al_right_full)
        al_left = al_left_full[start:end]
        al_right = al_right_full[start:end]
        # Now we need a metric of alignament quality. To start with, we will use basic number of matches metric
        matches_threshold = 0.5 # Should be paramether
        matches = 0
        for i in range(len(al_left)):
            if al_left[i] == al_right[i]:
                matches += 1
        matches_rate = matches / len(al_left)
        if matches_rate > matches_threshold and matches >= min_pairs:
            ptr = left_left
            for i in range(start):
                if al_left_full[i] != "-":
                    ptr += 1
            gaps = 0  # We count gaps not to skip it in original sequence
            left_edge = None
            right_edge = None
            for i in range(len(al_left)):
                # print(i)
                if al_left[i] == "-":
                    gaps += 1
                if al_left[i] == al_right[i]:
                    place = ptr + i - gaps
                    if left_edge is None:
                        left_edge = place
                    result[place] = "("
            ptr = right_right
            for i in range(start):
                if al_right_full[i] != "-":
                    ptr -= 1  # We go from the end, because we have reversed alignment
            gaps = 0
            for i in range(len(al_left)):
                if al_right[i] == "-":
                    gaps += 1
                if al_left[i] == al_right[i]:
                    place = ptr - i + gaps
                    if right_edge is None or place > right_edge:
                        right_edge = place
                    result[place] = ")"
            if left_edge is not None and right_edge is not None:
                edges.append((left_edge, right_edge))
        return ix
    ret = 0
    while ret != len(dot):
        ret = process_stem(ret)
    open = 0
    close = 0
    for el in result:
        if el == "(":
            open+=1
        elif el == ")":
            close +=1
    if open != close:
        sys.stderr.write("Brackets don't match on seq " + seq + " with dot " + dot)
    return seq, ''.join(result)


def dot2img(seq, dot):
    size = len(seq)
    codes = {'A': 32, 'C': 64, 'G': 96, 'U': 128, 'T': 128}
    stack = []
    mtrx = []
    for i in range(len(dot)):
        if dot[i] != ')':
            stack.append((i, dot[i]))
        else:
            while stack[-1][1] != '(':
                stack.pop()
            mtrx.append((i, stack[-1][0]))
            stack.pop()
    img = Image.new('L', (size, size), (0))
    pixels = np.array(img)
    for el in mtrx: 
        x, y = el
        pixels[y][x] = 255
    for i in range(len(seq)):
        pixels[i][i] = codes[seq[i]]
    return Image.fromarray(pixels)


def align(img):
    seq, dot = img2dot(img)
    seq, dot = align_sequence(seq, dot)
    return dot2img(seq, dot)
