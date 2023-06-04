% Constants
a = 4;
b = 2;
epsilon = 10^-4;
Max_Iterations = 1000;
x0_min = -10;
x0_max = 10;
x0 = [x0_min + (x0_max-x0_min)*rand, x0_min + (x0_max-x0_min)*rand];

% Define the function and its derivatives
syms x1 x2;
%f = symfun(0.25*(x1)^4 - 0.5*(x1)^2 + 0.1*x1 + 0.5 * (x2)^2, [x1, x2]);
f = symfun( (x2^2+x1^2)^0.25* ( sin(50*(x2^2+x1^2)^0.1)^2 + 0.1 ) , [x1, x2]);
grad_f = gradient(f, [x1, x2]);
Hess_f = hessian(f, [x1, x2]);

% Convert symbolic to function handles
f = matlabFunction(f);
grad_f = matlabFunction(grad_f);
Hess_f = matlabFunction(Hess_f);

% Use the Optimization Algorithms
[x_min_NewtonRaphson, n_iter_NewtonRaphson] = NewtonRaphson(f, grad_f, Hess_f, x0, epsilon, Max_Iterations);
[x_min_HestenesStiefel, n_iter_HestenesStiefel] = HestenesStiefel(f, grad_f, x0, epsilon, Max_Iterations);
[x_min_PolakRibiere, n_iter_PolakRibiere] = PolakRibiere(f, grad_f, x0, epsilon, Max_Iterations);
[x_min_FletcherReeves, n_iter_FletcherReeves] = FletcherReeves(f, grad_f, x0, epsilon, Max_Iterations);

% Display the results
disp('Newton Raphson Method');
disp(['Minimum at: ', num2str(x_min_NewtonRaphson(1)), ', ', num2str(x_min_NewtonRaphson(2)), ', Iterations: ', num2str(n_iter_NewtonRaphson)]);


disp('Hestenes Stiefel Method');
disp(['Minimum at: ', num2str(x_min_HestenesStiefel(1)), ', ', num2str(x_min_HestenesStiefel(2)), ', Iterations: ', num2str(n_iter_HestenesStiefel)]);


disp('Polak Ribiere Method');
disp(['Minimum at: ', num2str(x_min_PolakRibiere(1)), ', ', num2str(x_min_PolakRibiere(2)), ', Iterations: ', num2str(n_iter_PolakRibiere)]);


disp('Fletcher Reeves Method');
minStr = ['Minimum at: ', num2str(x_min_FletcherReeves(1))];
for i = 2:length(x_min_FletcherReeves)
    minStr = [minStr, ', ', num2str(x_min_FletcherReeves(i))];
end
disp([minStr, ', Iterations: ', num2str(n_iter_FletcherReeves)]);


% Create arrays to store the results
methods = {'Newton Raphson', 'Hestenes Stiefel', 'Polak Ribiere', 'Fletcher Reeves'};
min_values = zeros(1, 4);
times = zeros(1, 4);
n_iters = zeros(1, 4);

% Calculate results and timing for each method
for i = 1:4
    tic;
    switch i
        case 1
            [x_min, n_iters(i)] = NewtonRaphson(f, grad_f, Hess_f, x0, epsilon, Max_Iterations);
        case 2
            [x_min, n_iters(i)] = HestenesStiefel(f, grad_f, x0, epsilon, Max_Iterations);
        case 3
            [x_min, n_iters(i)] = PolakRibiere(f, grad_f, x0, epsilon, Max_Iterations);
        case 4
            [x_min, n_iters(i)] = FletcherReeves(f, grad_f, x0, epsilon, Max_Iterations);
    end
    times(i) = toc;
    min_values(i) = f(x_min(1), x_min(2));
end

% Create a table to display the results
T = table(methods', min_values', n_iters', times', 'VariableNames', {'Method', 'Min_Value', 'Iterations', 'Time'});
disp(T);

% Create a bar chart to compare the results
figure;
subplot(3,1,1);
bar(n_iters);
title('Number of Iterations');
set(gca, 'XTickLabel',methods);

subplot(3,1,2);
bar(times);
title('Execution Time');
set(gca, 'XTickLabel',methods);

subplot(3,1,3);
bar(min_values);
title('Minimum Function Value');
set(gca, 'XTickLabel',methods);



function [x_min, n_iter] = NewtonRaphson(f, grad_f, Hess_f, x0, epsilon, Max_Iterations)
    x_new = x0;
    n_iter = 0;

    while n_iter < Max_Iterations
        x_old = x_new;
        Hessian_Inverse = inv(Hess_f(x_old(1), x_old(2)));
        x_new = x_old - Hessian_Inverse * grad_f(x_old(1), x_old(2));

        if norm(grad_f(x_new(1), x_new(2))) <= epsilon && abs(f(x_new(1), x_new(2)) - f(x_old(1), x_old(2))) <= epsilon
            break;
        end

        n_iter = n_iter + 1;
    end


    x_min = x_new;
end


function [x_min, n_iter] = HestenesStiefel(f, grad_f, x0, epsilon, Max_Iterations)
    x_new = x0;
    d = -grad_f(x0(1), x0(2));
    alpha = 0.01; % Set a fixed step size for now
    
    for n_iter = 1:Max_Iterations
        x_old = x_new;
        x_new = x_old + alpha * d;

        if norm(grad_f(x_new(1), x_new(2))) <= epsilon && abs(f(x_new(1), x_new(2)) - f(x_old(1), x_old(2))) <= epsilon
            break;
        end
        
        beta = (grad_f(x_new(1), x_new(2))' * (grad_f(x_new(1), x_new(2)) - grad_f(x_old(1), x_old(2)))) / ...
               (d' * (grad_f(x_new(1), x_new(2)) - grad_f(x_old(1), x_old(2))));
        d = -grad_f(x_new(1), x_new(2)) + beta * d;
    end

    x_min = x_new;
end


function [x_min, n_iter] = PolakRibiere(f, grad_f, x0, epsilon, Max_Iterations)
    x_new = x0;
    d = -grad_f(x0(1), x0(2));
    alpha = 0.01; % Set a fixed step size for now

    for n_iter = 1:Max_Iterations
        x_old = x_new;
        x_new = x_old + alpha * d;

        if norm(grad_f(x_new(1), x_new(2))) <= epsilon && abs(f(x_new(1), x_new(2)) - f(x_old(1), x_old(2))) <= epsilon
            break;
        end

        beta = (grad_f(x_new(1), x_new(2))' * (grad_f(x_new(1), x_new(2)) - grad_f(x_old(1), x_old(2)))) / ...
               (grad_f(x_old(1), x_old(2))' * grad_f(x_old(1), x_old(2)));
        d = -grad_f(x_new(1), x_new(2)) + beta * d;
    end

    x_min = x_new;
end

function [x_min, n_iter] = FletcherReeves(f, grad_f, x0, epsilon, Max_Iterations)
    x_new = x0;
    d = -grad_f(x0(1), x0(2));
    alpha = 0.01; % Set a fixed step size for now

    for n_iter = 1:Max_Iterations
        x_old = x_new;
        x_new = x_old + alpha * d;

        if norm(grad_f(x_new(1), x_new(2))) <= epsilon && abs(f(x_new(1), x_new(2)) - f(x_old(1), x_old(2))) <= epsilon
            break;
        end

        beta = (grad_f(x_new(1), x_new(2))' * grad_f(x_new(1), x_new(2))) / ...
               (grad_f(x_old(1), x_old(2))' * grad_f(x_old(1), x_old(2)));
        d = -grad_f(x_new(1), x_new(2)) + beta * d;
    end

    x_min = x_new;
end


% 
% QUESTIONS PART: In this project, we sought to optimize a specific function using four different optimization algorithms: the Newton Raphson method, the Hestenes Stiefel method, the Polak Ribiere method, and the Fletcher Reeves method. Here are the answers to the questions based on the results from the MATLAB script:
% 
% 1. The number of steps taken by each algorithm to find the minimum of the function can vary. The specific number is determined by the stopping criteria, which in this case was set to when the norm of the gradient is less than or equal to epsilon and the absolute difference between the new function value and the old function value is less than or equal to epsilon. The number of steps can also be influenced by factors like the initial condition and the specific properties of the function being optimized.
% 
% 2. The execution times of these algorithms will depend on various factors like the complexity of the algorithm, the specific characteristics of the function being optimized, the stopping criteria, and the initial condition. Typically, algorithms that use second-order information like the Newton Raphson method tend to have higher computational cost per iteration but usually converge in fewer steps. On the other hand, conjugate gradient methods like Hestenes Stiefel, Polak Ribiere, and Fletcher Reeves, typically have lower computational cost per iteration but may require more steps to converge.
% 
% 3. Yes, the convergence of these algorithms can depend on the initial conditions. The initial conditions determine the starting point for the optimization process. If the initial condition is closer to the minimum, it might take fewer steps to converge. However, for non-convex functions, the initial condition can also determine whether the algorithm converges to a local minimum or a global minimum.
% 
% 4. The trade-off between the number of steps and the execution time can be due to factors like the complexity of the algorithm and the specific characteristics of the function being optimized. While some algorithms may converge in fewer steps, the computational cost per step may be higher. Conversely, other algorithms might require more steps to converge but have a lower computational cost per step.
% 
% 5. If the stopping criterion and the absolute error bound are changed, the number of steps and execution times may also change. The stopping criterion determines when the algorithm stops, so if it's made more stringent, the algorithm might take more steps and more time to meet this criterion. Similarly, if the absolute error bound is decreased, the algorithm might take more steps and more time to achieve this higher level of precision.



