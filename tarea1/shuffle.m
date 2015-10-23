function B=Shuffle(A)
% function B=Shuffle(A)
%
%  The function randomly permute the elements of a vector. If A is a matrix
%  then the elements are permuted by row
%
% Input parameters:
%   A: Vector or array.
%
% Output parameters:
%   B: Vector or array with the elements of A but in different order
%
% Example:
%   A=rand(5,3);
%   B=Shuffle(A)
% 
% Program created by R.Salas and R.Torres
% Last revision: 2005-02-21 (R.Salas)

I=randperm(length(A'));
B=A(I,:);
