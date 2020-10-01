import numpy as np
import plaidml
import plaidml.edsl as edsl
import plaidml.op as op
import plaidml.exec as plaidml_exec

a = np.random.rand(10)
b = np.random.rand(10)

dtype = plaidml.DType.FLOAT32

A = edsl.Tensor(edsl.LogicalShape(dtype, a.shape))
B = edsl.Tensor(edsl.LogicalShape(dtype, b.shape))

#PlaidML
Asm = 2 * op.sum(A, axis=0)
C = Asm * B

#Numpy
asm = 2 * sum(a)
c = asm * b

program = edsl.Program('program', [Asm, C])
binder = plaidml_exec.Binder(program)
executable = binder.compile()

binder.input(A).copy_from_ndarray(a)
binder.input(B).copy_from_ndarray(b)

executable.run()

print("Asm, expected vs recieved")
print(asm)
print(binder.output(Asm).as_ndarray())

print("C, expected vs recieved")
print(c)
print(binder.output(C).as_ndarray())
