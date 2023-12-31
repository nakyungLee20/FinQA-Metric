from copy import deepcopy
import re


# An expression tree node
class Et:
    # Constructor to create a node
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


# Returns root of constructed tree for given postfix expression
def construct_exp_tree(postfix):
    stack = []

    # Traverse through every character of input expression
    for char in postfix:

        # if operand, simply push into stack
        if char not in ["+", "-", "*", "/", "**", ">", "^", "T"]:
            t = Et(char)
            stack.append(t)
        # Operator
        else:
            # Pop two top nodes
            t = Et(char)
            t1 = stack.pop()
            t2 = stack.pop()

            # make them children
            t.right = t1
            t.left = t2

            # Add this subexpression to stack
            stack.append(t)
    # Only element  will be the root of expression tree
    t = stack.pop()
    return t


def infix_to_postfix(expression):
    stack = []
    postfix = []
    # operator = ['+', '-', '*', '/', '**', '^', '>', 'T']
    operator = {"+": 0, "-": 0, ">": 0, "T": 0, "*": 1, "/": 1, "**": 2, "^": 2 }

    for char in expression:
        if char in operator:  # Operator
            while len(stack) > 0 and stack[-1] not in ['(', '['] and operator[stack[-1]] >= operator[char]:
                postfix.append(stack.pop())
            stack.append(char)
        elif char in ['(', '[']:
            stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(':
                postfix.append(stack.pop())
            stack.pop()  # Discard '('
        elif char == "]":
            while stack and stack[-1] != '[':
                postfix.append(stack.pop())
            stack.pop()
        else:   # Operand
            postfix.append(char)

    while stack:
        postfix.append(stack.pop())

    return postfix


def infix_to_prefix(expression):
    stack = []
    prefix = []
    operator = {"+": 0, "-": 0, ">": 0, "T": 0, "*": 1, "/": 1, "**": 2, "^": 2}
    expression = expression[::-1]  # Reverse the expression

    for char in expression:
        if char in operator:  # Operator
            while len(stack)>0 and stack[-1] not in [")", "]"] and operator[stack[-1]] > operator[char]:
                prefix.append(stack.pop())
            stack.append(char)
        elif char in [')', ']']:
            stack.append(char)
        elif char == '(':
            while stack and stack[-1] != ')':
                prefix.append(stack.pop())
            stack.pop()  # Discard ')'
        elif char == '[':
            while stack and stack[-1] != ']':
                prefix.append(stack.pop())
            stack.pop()  # Discard ']'
        else:   # Operand
            prefix.append(char)

    while stack:
        prefix.append(stack.pop())

    prefix = prefix[::-1]  # Reverse the prefix expression
    return prefix


def from_infix_to_postfix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "**": 2, "^": 2, ">": 0, "T": 0}
    for e in expression:
        if e in ["(", "["]:
            st.append(e)
        elif e == ")":
            c = st.pop()
            while c != "(":
                res.append(c)
                c = st.pop()
        elif e == "]":
            c = st.pop()
            while c != "[":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in ["(", "["] and priority[e] <= priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())

    # print(res)
    return res


def from_infix_to_prefix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "**": 2, "^": 2, ">": 0, "T": 0}
    expression = deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()

    return res


def out_expression_list(test, output_lang, num_list, num_stack=None):
    max_index = output_lang.n_words
    res = []
    for i in test:
        # if i == 0:
        #     return res
        if i < max_index - 1:
            idx = output_lang.index2word[i]
            if idx[0] == "N":
                if int(idx[1:]) >= len(num_list):
                    return None
                res.append(num_list[int(idx[1:])])
            else:
                res.append(idx)
        else:
            pos_list = num_stack.pop()
            c = 1 # 兜底逻辑
            if (len(num_list) > 0):
                c = num_list[0]
            if (len(pos_list) > 0):
                c = num_list[pos_list[0]]
            res.append(c)
    return res


def compute_postfix_expression(post_fix):
    st = list()
    operators = ["+", "-", "**", "*", "/", ">", "^", "T"]

    for p in post_fix:
        if p not in operators:
            pos = re.search("\d+\(", p)
            if pos:
                st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
            elif p[-1] == "%":
                    st.append(float(p[:-1]) / 100)
            else:
                st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if a == 0:
                return None
            st.append(b / a)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(b - a)
        elif p == "**" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a ** b)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a ** b)
        elif p == ">" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a)
        elif p == "T" and len(st) > 1:
            a = st.pop()
            st.append(a)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None


def compute_prefix_expression(pre_fix):
    st = list()
    operators = ["+", "-", "**", "*", "/", ">", "^", "T"]
    pre_fix = deepcopy(pre_fix)
    pre_fix.reverse()
    for p in pre_fix:
        if p not in operators:
            pos = re.search("\d+\(", p)
            if pos:
                st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
            elif p[-1] == "%":
                st.append(float(p[:-1]) / 100)
            else:
                st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if b == 0:
                return None
            st.append(a / b)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a - b)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
                return None
            st.append(a ** b)
        elif p == "**" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
                return None
            st.append(a ** b)
        elif p == ">" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a)
        elif p == "T" and len(st) > 1:
            a = st.pop()
            st.append(a)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None

