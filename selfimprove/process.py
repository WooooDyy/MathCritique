import re
import pdb

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _space(string):
    if " " in string: 
        matches = string.split(" ")
        matches = [m for m in matches if m != ""]
        answer = ""
        for idx, match in enumerate(matches):
            answer += match
            if "\\" in match and idx+1 < len(matches) and matches[idx+1][0].isalpha():
                answer += " "
        string = answer
    return string

def answer_process(string):
    if string == None:
        return None

    string = str(string).strip()

    ## \\$
    if '\\\\' in string:
        string = string.replace("\\\\", "\\")
    if '\\$' in string:
        string = string.replace("\\$", "")
    ## ,\\!
    if ',\\!' in string:
        string = string.replace(",\\!", "")
    if '\\!' in string:
        string = string.replace("\\!", "")
    ## \\cdot
    if "\\cdot" in string:
        string = string.replace("\\cdot", "")
    ## \\%
    if "\\%" in string:
        string = string.replace("\\%", "%")
        
    # \n
    if "\n" in string:
        string = string.replace("\n", "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")
    
    if '\\dfrac' in string:
        string = string.replace('\\dfrac', '\\frac')

    if '\\tfrac' in string:
        string = string.replace('\\tfrac', '\\frac')

    string = string.replace("\\left", "").strip()
    string = string.replace("\\right", "").strip()

    string = string.replace("^{\\circ}", "").strip()
    string = string.replace("^\\circ", "").strip()

    if "{,}" in string:
        string = string.replace("{,}", "")
    if "\\," in string:
        string = string.replace("\\,", "")

    if "\\text{" in string:
        match = re.search(r'\\text{(.*?)}', string)
        answer = re.sub(r'\\text\{', '', re.sub(r'\}', '', match[0]))
        if answer == " or " or answer == ", and ":
            string = string.replace(match[0], ",")
        elif answer[0] == ' ':
            string = string.replace(match[0], "")
        else:
            string = string.replace(match[0], answer)

    if "\\mbox{" in string:
        match = re.search(r'\\mbox{(.*?)}', string)
        answer = re.sub(r'\\mbox\{', '', re.sub(r'\}', '', match[0]))
        if answer[0] == ' ':
            string = string.replace(match[0], "")
        else:
            string = string.replace(match[0], answer)    

    if "\\textnormal{" in string:
        match = re.search(r'\\textnormal{(.*?)}', string)
        answer = re.sub(r'\\textnormal\{', '', re.sub(r'\}', '', match[0]))
        if answer[0] == ' ':
            string = string.replace(match[0], "")
        else:
            string = string.replace(match[0], answer)    
    
    if "\\mathbf{" in string:
        match = re.search(r'\\mathbf{(.*?)}', string)
        answer = re.sub(r'\\mathbf\{', '', re.sub(r'\}', '', match[0]))
        if answer[0] == ' ':
            string = string.replace(match[0], "")
        else:
            string = string.replace(match[0], answer)   

    string = _space(string)

    string = _fix_fracs(string)

    string = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", string)
    string = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", string)

    string = re.sub(r"\\tan(-?[0-9.a-zA-Z]+)", r"\\tan{\1}", string)
    string = re.sub(r"\\tan\s+(\w+)$", r"\\tan{\1}", string)

    string = _fix_a_slash_b(string)

    try:
        string = str(string).strip()
        if string[0] == '.' and any(c.isdigit() for c in string[1:]):
            string = '0' + string

        if string[0] == '(' and string[2] == ')' and string[1].isalpha():
            string = string.replace(string, string[1])     

        string = str(string).strip()
        if string[-1] == "," or string[-1] == "." or string[-1] == "\\":
            string = string[:-1]
        string = str(string).strip()
    except:
        pass
    
    if "_" in string:
        pos = string.find(r'_')
        if string[pos+1] != '{':
            string = string[:pos+1] + '{' + string[pos+1:] + '}'
    
    if "pmatrix" in string:
        string = string.replace("\\phantom-", "")
        string = string.replace("\\","\\\\")
        string = string.replace("\\\\begin","\\begin")
        string = string.replace("\\\\end","\\end")
        string = string.replace("\\\\frac", "\\frac")
        string = string.replace("\\\\sqrt", "\\sqrt")
    string = str(string).strip()

    string = _space(string)
    return str(string)


def extract_boxed_content(text):
    start_pos = text.rfind(r'\boxed{')
    if start_pos == -1:
        return None
    cnt = 1
    end_pos = 0
    for i in range(start_pos + len(r'\boxed{'), len(text)):
        if text[i] == '{':
            cnt += 1
        elif text[i] == '}':
            cnt -= 1
        if cnt == 0:
            end_pos = i
            break
    if cnt == 0:
        return str(text[start_pos + len(r'\boxed{'): end_pos])
    else:
        return None

if __name__ == "__main__":
    from datasets import load_dataset
    import pdb
    dataset = load_dataset("lighteval/MATH", "all", split = "train", trust_remote_code=True)
    for data in dataset:
        text = data["solution"]
        text1 = extract_boxed_content(text)
        text2 = answer_process(text1)
        if text2 == None or len(text2) == 0:
            pdb.set_trace()