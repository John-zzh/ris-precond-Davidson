
python PySCF_grad.py -x *.xyz -b def2-svp -f pbe0 -df True -M 4000 -grad true -chk true
# python PySCF_grad.py -x *.xyz -b def2-tzvp -f pbe0 -df True -M 4000 -grad true -chk true > $(basename *.xyz .xyz).out 2>&1
