clear
% close

proxOps = ProximalOperators();
arr0 = [-3 -2 -1 0 1 2 3];
arr1 = [-3 -2 -1; -1 0 1;];
gamma = 1;
l = -1;
r = 1;

ret0 = proxOps.proxL1Norm(arr0, gamma);
ret1 = proxOps.proxL1Norm(arr1, gamma);
ret2 = proxOps.proxNuclearNorm(arr1, gamma);
ret3 = proxOps.proxBoxConstraint(arr0, l, r);
ret4 = proxOps.proxBoxConstraint(arr1, l, r);

disp(ret0)
disp(ret1)
disp(ret2)
disp(ret3)
disp(ret4)
