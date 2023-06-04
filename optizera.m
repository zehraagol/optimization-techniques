clear
clc

% Constants
epsilon = 10^-4;
Max_Iterations = 1000;
x0_min = -512;
x0_max = 512;
x0 = [x0_min + (x0_max-x0_min)*rand, x0_min + (x0_max-x0_min)*rand];

% Define the function and its derivatives
syms x1 x2;
f = symfun(-(x2 + 47) * sin(sqrt(abs(x2 + x1 / 2 + 47))) - x1 * sin(sqrt(abs(x1 - (x2 + 47)))), [x1, x2]);
grad_f = gradient(f, [x1, x2]);
Hess_f = hessian(f, [x1, x2]);

% Convert symbolic to function handles
f = matlabFunction(f);
grad_f = matlabFunction(grad_f);
Hess_f = matlabFunction(Hess_f);

% Create arrays to store the results
methods = {'Newton Raphson', 'Hestenes Stiefel', 'Polak Ribiere', 'Fletcher Reeves'};
min_values = zeros(50, 4);
times = zeros(50, 4);
n_iters = zeros(50, 4);

% Run the optimization algorithms 50 times
for run = 1:100
    disp(['Run ', num2str(run)]);
    x0 = [x0_min + (x0_max-x0_min)*rand, x0_min + (x0_max-x0_min)*rand];

    % Calculate results and timing for each method
    for i = 1:4
        tic;
        switch i
            case 1
                [x_min, n_iters(run, i)] = NewtonRaphson(f, grad_f, Hess_f, x0, epsilon, Max_Iterations, x0_min, x0_max);
            case 2
                [x_min, n_iters(run, i)] = HestenesStiefel(f, grad_f, x0, epsilon, Max_Iterations, x0_min, x0_max);
            case 3
                [x_min, n_iters(run, i)] = PolakRibiere(f, grad_f, x0, epsilon, Max_Iterations, x0_min, x0_max);
            case 4
                [x_min, n_iters(run, i)] = FletcherReeves(f, grad_f, x0, epsilon, Max_Iterations, x0_min, x0_max);
        end
        times(run, i) = toc;
        min_values(run, i) = f(x_min(1), x_min(2));
    end
end

% Calculate the average results and timing
avg_min_values = mean(min_values);
avg_times = mean(times);
avg_n_iters = mean(n_iters);

% Create a table to display the average results
T = table(methods', avg_min_values', avg_n_iters', avg_times', 'VariableNames', {'Method', 'Min_Value', 'Iterations', 'Time'});
disp(T);

% Create a bar chart to compare the average results
figure;
subplot(3,1,1);
bar(avg_n_iters);
title('Number of Iterations');
set(gca, 'XTickLabel', methods);

subplot(3,1,2);
bar(avg_times);
title('Execution Time');
set(gca, 'XTickLabel', methods);

subplot(3,1,3);
bar(avg_min_values);
title('Minimum Function Value');
set(gca, 'XTickLabel', methods);


% Generate a meshgrid of x1 and x2 values
x1_values = linspace(x0_min, x0_max, 100);
x2_values = linspace(x0_min, x0_max, 100);
[X1, X2] = meshgrid(x1_values, x2_values);

% Evaluate the function at each grid point
Z = f(X1, X2);

% Create a surface plot of the function
figure;
surf(X1, X2, Z);
xlabel('x1');
ylabel('x2');
zlabel('f(x1, x2)');
title('Function');

x = 1:size(min_values, 1);  % x-axis values are the row indices

% Plot the first line graph
y1 = min_values(:, 1);      % y-axis values for the first column
figure;                    % create a new figure window
plot(x, y1)
title('Min values of Newton Raphson')

% Plot the second line graph
y2 = min_values(:, 2);      % y-axis values for the second column
figure;                    % create another figure window
plot(x, y2)
title('Min values of Hestenes Stiefel')

% Plot the third line graph
y3 = min_values(:, 3);      % y-axis values for the third column
figure;                    % create another figure window
plot(x, y3)
title('Min values of Polak Ribiere')

% Plot the fourth line graph
y4 = min_values(:, 4);      % y-axis values for the fourth column
figure;                    % create another figure window
plot(x, y4)
title('Min values of Fletcher Reeves')



function [x_min, n_iter] = NewtonRaphson(f, grad_f, Hess_f, x0, epsilon, Max_Iterations, x0_min, x0_max)
    x_new = x0;
    n_iter = 0;

    while n_iter < Max_Iterations
        x_old = x_new;
        Hessian_Inverse = inv(Hess_f(x_old(1), x_old(2)));
        x_new = x_old - Hessian_Inverse * grad_f(x_old(1), x_old(2));
        x_new = max(x_new, x0_min);
        x_new = min(x_new, x0_max);


        if norm(grad_f(x_new(1), x_new(2))) <= epsilon && abs(f(x_new(1), x_new(2)) - f(x_old(1), x_old(2))) <= epsilon
            break;
        end

        n_iter = n_iter + 1;
    end

    x_min = x_new;
end

function [x_min, n_iter] = HestenesStiefel(f, grad_f, x0, epsilon, Max_Iterations, x0_min, x0_max)
    x_new = x0;
    d = -grad_f(x0(1), x0(2));
    alpha = 0.01; % Set a fixed step size for now

    for n_iter = 1:Max_Iterations
        x_old = x_new;
        x_new = x_old + alpha * d;
        x_new = max(x_new, x0_min);
        x_new = min(x_new, x0_max);


        if norm(grad_f(x_new(1), x_new(2))) <= epsilon && abs(f(x_new(1), x_new(2)) - f(x_old(1), x_old(2))) <= epsilon
            break;
        end

        beta = (grad_f(x_new(1), x_new(2))' * (grad_f(x_new(1), x_new(2)) - grad_f(x_old(1), x_old(2)))) / ...
            (d' * (grad_f(x_new(1), x_new(2)) - grad_f(x_old(1), x_old(2))));
        d = -grad_f(x_new(1), x_new(2)) + beta * d;
    end

    x_min = x_new;
end

function [x_min, n_iter] = PolakRibiere(f, grad_f, x0, epsilon, Max_Iterations, x0_min, x0_max)
    x_new = x0;
    d = -grad_f(x0(1), x0(2));
    alpha = 0.01; % Set a fixed step size for now

    for n_iter = 1:Max_Iterations
        x_old = x_new;
        x_new = x_old + alpha * d;
        x_new = max(x_new, x0_min);
        x_new = min(x_new, x0_max);

        if norm(grad_f(x_new(1), x_new(2))) <= epsilon && abs(f(x_new(1), x_new(2)) - f(x_old(1), x_old(2))) <= epsilon
            break;
        end

        beta = (grad_f(x_new(1), x_new(2))' * (grad_f(x_new(1), x_new(2)) - grad_f(x_old(1), x_old(2)))) / ...
            (grad_f(x_old(1), x_old(2))' * grad_f(x_old(1), x_old(2)));
        d = -grad_f(x_new(1), x_new(2)) + beta * d;
    end

    x_min = x_new;
end

function [x_min, n_iter] = FletcherReeves(f, grad_f, x0, epsilon, Max_Iterations, x0_min, x0_max)
    x_new = x0;
    d = -grad_f(x0(1), x0(2));
    alpha = 0.01; % Set a fixed step size for now

    for n_iter = 1:Max_Iterations
        x_old = x_new;
        x_new = x_old + alpha * d;
        x_new = max(x_new, x0_min);
        x_new = min(x_new, x0_max);

        if norm(grad_f(x_new(1), x_new(2))) <= epsilon && abs(f(x_new(1), x_new(2)) - f(x_old(1), x_old(2))) <= epsilon
            break;
        end

        beta = (grad_f(x_new(1), x_new(2))' * grad_f(x_new(1), x_new(2))) / ...
            (grad_f(x_old(1), x_old(2))' * grad_f(x_old(1), x_old(2)));
        d = -grad_f(x_new(1), x_new(2)) + beta * d;
    end

    x_min = x_new;
end
