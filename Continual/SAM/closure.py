def out_f(s):
    message = s+', hello'
    def in_f():
        print(message)
    return in_f

f=out_f('sejun')
f()
g=out_f('heize')
g()
f()