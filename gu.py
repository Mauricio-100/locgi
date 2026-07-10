import kenlm

model = kenlm.Model("gu.binary")
print(model.score("આ એક વાક્ય છે."))
