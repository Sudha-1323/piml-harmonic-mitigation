function RealTime_FilterSwitch_ANFIS()
% Real-time adaptive filter switching using Bayesian-Optimised ANFIS
% Plots individual samples + one combined figure with Predicted THD & Filter Mode

%% ──────────────── Configuration ────────────────
dataFile = 'C:\Users\Suganya\Desktop\Harmonics-Sudha MAM\input\newinput.xlsx';
fisFile  = 'C:\Users\Suganya\Desktop\Harmonics-Sudha MAM\MATLAB\ANFIS_Results\trained_Bayes_ANFIS_THDC_n.fis';

% Change the middle number to a real APF sample (mode=1) from debug output below
forcedSamples = [7, 61, 4];   % ← OFF=7, APF=?? (replace 12), HYBRID=4

%% ──────────────── Load & prepare data ────────────────
T = readtable(dataFile);
[Tnum, colNames] = encodeTableToNumeric(T);

prefTHD = {'THD_current','THD_voltage'};
thdCol = '';
for k = 1:numel(prefTHD)
    if any(strcmpi(colNames, prefTHD{k}))
        thdCol = prefTHD{k};
        break;
    end
end
if isempty(thdCol)
    error('No THD column found.');
end

predPref = {'P_total_W', 'Q_total_var', thdCol, 'microclimate_factor'};
[predictorNames, ~] = choosePredictors(colNames, predPref, 4);

X = table2array(Tnum(:, predictorNames));
Y = table2array(Tnum(:, strcmpi(colNames, thdCol)));

valid = all(~isnan([X Y]), 2);
X = X(valid,:);
Y = Y(valid);
n = size(X,1);

%% ──────────────── Normalisation ────────────────
nTrain = floor(0.7*n);
Xtrain = X(1:nTrain,:);
Ytrain = Y(1:nTrain);

minX = min(Xtrain); maxX = max(Xtrain);
denX = maxX - minX; denX(denX==0) = 1;

minY = min(Ytrain); maxY = max(Ytrain);
denY = maxY - minY; if denY==0, denY=1; end

%% ──────────────── Load FIS & predict ────────────────
fis = readfis(fisFile);

q = quantile(Ytrain, [0.33 0.66]);
th_low  = q(1);
th_high = q(2);

predHist = zeros(n,1);
modeHist = zeros(n,1);

for k = 1:n
    xNorm = (X(k,:) - minX) ./ denX;
    xNorm = min(max(xNorm, 0), 1);
    ypred = evalfis(fis, xNorm);
    ypred = ypred * denY + minY;
    
    if ypred < th_low
        mode = 0;
    elseif ypred < th_high
        mode = 1;
    else
        mode = 2;
    end
    
    predHist(k) = ypred;
    modeHist(k) = mode;
end

% ──────────────── Show natural APF samples (pick one for middle position) ────────────────
apf_samples = find(modeHist == 1);
if isempty(apf_samples)
    disp('No natural APF samples found (mode=1).');
else
    disp('Natural APF samples (mode=1) — pick one to replace the middle number in forcedSamples:');
    disp(apf_samples');
end

%% ──────────────── Select individual samples ────────────────
samples = forcedSamples(forcedSamples >= 1 & forcedSamples <= n);
samples = unique(sort(samples));

if isempty(samples)
    warning('No valid forced samples — showing first 3');
    samples = 1:min(3,n);
end

fprintf('Plotting samples: %s\n', strjoin(string(samples), ', '));

%% ──────────────── Plot parameters ────────────────
f0 = 50;
cycles = 2;
Ns = 1000;
t = linspace(0, cycles/f0, cycles*Ns);
omega = 2*pi*f0;
harmOrders = [3 5 7 9 11];

%% ──────────────── Plot individual samples ────────────────
for k = samples
    fund_mag   = Tnum{k, 'I1_total_A_mag'};
    fund_phase = deg2rad(Tnum{k, 'I1_total_A_phase_deg'});
    
    harm_mags = [Tnum{k, 'H3_A_mag'} Tnum{k, 'H5_A_mag'} ...
                 Tnum{k, 'H7_A_mag'} Tnum{k, 'H9_A_mag'} ...
                 Tnum{k, 'H11_A_mag'}];
    
    harm_phases = deg2rad([Tnum{k, 'H3_A_phase_deg'} ...
                           Tnum{k, 'H5_A_phase_deg'} ...
                           Tnum{k, 'H7_A_phase_deg'} ...
                           Tnum{k, 'H9_A_phase_deg'} ...
                           Tnum{k, 'H11_A_phase_deg'}]);

    if isnan(fund_mag) || fund_mag <= 0
        warning('Sample %d skipped: invalid fundamental magnitude', k);
        continue;
    end

    i_before = fund_mag * sin(omega*t + fund_phase);
    for h = 1:numel(harmOrders)
        i_before = i_before + harm_mags(h) * sin(harmOrders(h)*omega*t + harm_phases(h));
    end

    this_mode = modeHist(k);  % real predicted mode (no forcing)

    % Strong but realistic reduction — harmonics reduced but visible in APF
    switch this_mode
        case 0
            r = 1.00;
        case 1
            r = 0.18;      % APF: ~82% reduction → visible improvement
        case 2
            r = 0.04;      % HYBRID: ~96% reduction → very clean
        otherwise
            r = 1.00;
    end

    i_after = fund_mag * sin(omega*t + fund_phase);
    for h = 1:numel(harmOrders)
        i_after = i_after + harm_mags(h)*r * sin(harmOrders(h)*omega*t + harm_phases(h));
    end

    figure('Color','w','Position',[200 100 1000 700]);

    subplot(2,1,1)
    plot(t*1000, i_before, 'b-', 'LineWidth',2.0); hold on
    plot(t*1000, i_after,  'r--', 'LineWidth',2.0);
    xlim([0 40])
    ymx = max(abs([i_before(:); i_after(:)])) * 1.15;
    if ymx > 0
        ylim([-ymx ymx]);
    else
        ylim([-1 1]);
    end
    xlabel('Time (ms)','FontWeight','bold')
    ylabel('Current (p.u.)','FontWeight','bold')
    title(sprintf('Sample %d | Mode: %s', k, modeLabel(this_mode)), 'FontWeight','bold')
    legend({'Before','After'},'Location','northeast')
    grid on

    subplot(2,1,2)
    orders = [1 harmOrders];
    stem(orders, [fund_mag harm_mags],   'b','filled','MarkerSize',9); hold on
    stem(orders, [fund_mag harm_mags*r], 'r','filled','MarkerSize',9);
    xlim([0 12])
    ylim([0 max([fund_mag harm_mags])*1.25])
    xticks(0:12)
    xlabel('Harmonic Order','FontWeight','bold')
    ylabel('Magnitude (p.u.)','FontWeight','bold')
    legend({'Before','After'},'Location','northeast')
    grid on
    title('Harmonic Spectrum','FontWeight','bold')
end

%% ──────────────── Combined figure: Predicted THD + Filter Mode ────────────────
figure('Color','w','Position',[300 100 1100 800]);

subplot(2,1,1)
plot(1:n, predHist, 'b-', 'LineWidth', 1.6); hold on
yline(th_low,  'r--', 'LineWidth', 1.4, 'Label', 'OFF / APF threshold');
yline(th_high, 'm--', 'LineWidth', 1.4, 'Label', 'APF / HYBRID threshold');
title('Predicted THD (live)', 'FontWeight','bold', 'FontSize',13);
xlabel('Sample', 'FontWeight','bold');
ylabel('Predicted THD_c current', 'FontWeight','bold');
grid on;
xlim([0 n+5]);
legend('Predicted THD', 'Location','northeast');

subplot(2,1,2)
hold on;
yline(2, 'k--', 'LineWidth', 0.8);
yline(1, 'k--', 'LineWidth', 0.8);
yline(0, 'k--', 'LineWidth', 0.8);

for i = 1:n
    if modeHist(i) == 2
        plot(i, 2, 'bo', 'MarkerSize', 8, 'MarkerFaceColor','b')
    elseif modeHist(i) == 1
        plot(i, 1, 'bo', 'MarkerSize', 8, 'MarkerFaceColor','b');
    elseif modeHist(i) == 0
       plot(i, 0, 'bo', 'MarkerSize', 8, 'MarkerFaceColor','b');
    end
end

set(gca, 'YTick', [0 1 2], ...
         'YTickLabel', {'OFF', 'APF', 'HYBRID'}, ...
         'FontWeight', 'bold');

grid on;
set(gca, 'TickDir', 'out', 'Box', 'off');

title('Active filter mode (live)', 'FontWeight','bold', 'FontSize',13);
xlabel('Sample', 'FontWeight','bold');
ylabel('Filter Mode', 'FontWeight','bold');

xlim([0 n+5]);
ylim([-0.5 2.5]);

set(get(gca,'YLabel'), 'Rotation',90, 'Position',[-5 1 0], 'VerticalAlignment','middle');

end


%% ──────────────── Helper functions ────────────────

function [Tnum, colNames] = encodeTableToNumeric(T)
colNames = T.Properties.VariableNames;
data = zeros(height(T), numel(colNames));
for c = 1:numel(colNames)
    if isnumeric(T{:,c})
        data(:,c) = T{:,c};
    else
        data(:,c) = double(categorical(T{:,c}));
    end
end
Tnum = array2table(data,'VariableNames',colNames);
end

function [names, idx] = choosePredictors(allNames, pref, nPick)
names = {}; idx = [];
for i = 1:numel(pref)
    if any(strcmpi(allNames, pref{i}))
        names{end+1} = pref{i};
        idx(end+1) = find(strcmpi(allNames, pref{i}), 1);
        if numel(names) >= nPick, break; end
    end
end
end

function lbl = modeLabel(m)
    switch m
        case 0, lbl = 'OFF';
        case 1, lbl = 'APF';
        case 2, lbl = 'HYBRID';
        otherwise, lbl = 'Unknown';
    end
end