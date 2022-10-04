import re


def calcul_simil(ori_sent, smm_sent):
    ori_sent_tokens = ori_sent.split(" ")
    smm_sent_tokens = smm_sent.split(" ")

    if ori_sent == '':
        return {'result': '원문이 없습니다'}
    if smm_sent == '':
        return {'result': '요약문이 없습니다'}

    cnt=0
    for smm_sent_token in smm_sent_tokens:
        if smm_sent_token in ori_sent_tokens:
            cnt+=1
    similarity = cnt/len(smm_sent_tokens)
    return similarity

def calcul_len(ori_sent,smm_sent):
    return len(smm_sent)/len(ori_sent)

def remove_html(prompt):
    #html 태그 제거
    result = re.sub(r'<[\\]{0,1}.*?>','',prompt)
    return result

def remove_latex(prompt):
    #mathrm 표현 제거
    p = re.compile(r'(\\(?:mathrm)(?=((?:{[^}]*}|\[[^]]*])*))\2)[^\S\n]*(?=\S)')
    res1 = p.sub(r'\2', prompt)
    #중괄호{} 제거
    aa = re.findall(r'\{.*?\}', res1)
    s = res1
    for w in aa:
        s = s.replace(w,w[1:-1])
    #'\' 이스케이프 문자와 인접한 소괄호() 제거
    res2 = s
    aa = re.findall(r'\\\(.*?\\\)', res2)
    ss = res2
    for w in aa:
        ss = ss.replace(w,w[2:-2])
    return ss

def split_textline(text, delimiter='다.'):
    ori_tls = []
    tst = text.split(delimiter)
    for i, l in enumerate(tst):

        if i < len(tst)-1:
            ori_tls.append(l+delimiter)
        elif l=='':
            pass
        else:
            ori_tls.append(l)
    return ori_tls
