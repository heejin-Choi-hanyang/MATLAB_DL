load_folder = 'Data_gesture';
sessions = 5;
trials = 20;
channels = 3;
fs = 1000;
classes = 8;
winLen = round(0.2 * fs);    % 200ms windows
overlap = round(0.18 * fs);  % 90% overlap (180ms)
hop = winLen - overlap;

fftLen = 256;
idx2label = {'forward','backward','up','down','left','right','stop','NaN'};

test_sig = zeros(fs,1);
[~, f, t_, ps] = spectrogram(test_sig, winLen, overlap, fftLen, fs);
fmin = 20;
fmax = 450;
idx_range = (f>fmin) & (f < fmax);
f_range = f(idx_range);
freq_bins = numel(f_range);
time_frames = numel(t_);

epsilon = 1e-6;
rng(42);

samples_total = sessions * trials * classes;
X_stft = zeros(freq_bins, time_frames, channels, samples_total); % [freq, time, ch, sample]
y = zeros(samples_total, 1);
sample_idx = 1;
for sess = 1:sessions
    label_list = readtable(strcat('gesture_list_session_', num2str(sess), '.csv'), 'FileEncoding', 'UTF-8');
    for trial = 1:trials
        for gesture = 1:classes
            mat_file = fullfile(load_folder, ['session' num2str(sess)], ['trial' num2str(trial)], ...
                ['gesture' num2str(label_list.word(gesture + (trial-1)*classes)) '.mat']);
            if ~isfile(mat_file)
                warning('File not found: %s', mat_file);
                continue;
            end
            S = load(mat_file);
            signal = S.get_data.save_data; % [N x 3]

            stft_tensor = zeros(freq_bins, time_frames, channels);
           
            for ch = 1:channels
                sig_ch = signal(:, ch);
                [~, ~, ~, ps] = spectrogram(sig_ch, winLen, overlap, fftLen, fs);
                ps_db = 10 * log10(abs(ps)); % dB-scaled spectrogram
                stft_tensor(:,:,ch) = ps_db(idx_range, :);
            end


            tmp = stft_tensor;
            mu = mean(tmp, [1 2]); sigma = std(tmp, 0, [1 2]);

            sigma(sigma < epsilon) = 1;
            stft_tensor = (tmp - mu) ./ sigma;

            X_stft(:, :, :, sample_idx) = stft_tensor;
           
            y(sample_idx) = label_list.word(gesture + (trial-1)*classes);
            sample_idx = sample_idx + 1;
        end
    end
end


X_stft = X_stft(:,:,:,1:sample_idx-1);
y = y(1:sample_idx-1);


tabulate(y)

disp(['전체 샘플 갯수: ', num2str(size(X_stft,4))]);

% Dataset Split: Cross-validation (Leave-one-session-out)
inputSize = [freq_bins, time_frames, 1];
numClasses = classes;

layers_ch1 = [
    imageInputLayer(inputSize, 'Name', 'input1', 'Normalization', 'none')

    convolution2dLayer([5 5], 4, 'Padding','same', 'Name','conv1_ch1')
    batchNormalizationLayer('Name','bn1_ch1')
    reluLayer('Name','relu1_ch1')
    maxPooling2dLayer([2 2], 'Stride', [2 2], 'Name','pool1_ch1')

    dropoutLayer(0.3, 'Name','dropout1_ch1')

    convolution2dLayer([3 3], 16, 'Padding','same', 'Name','conv2_ch1')
    batchNormalizationLayer('Name','bn2_ch1')
    reluLayer('Name','relu2_ch1')
    maxPooling2dLayer([2 2],'Stride', [2 2], 'Name','pool2_ch1')

    dropoutLayer(0.3, 'Name','dropout2_ch1')];

layers_ch2 = [
    imageInputLayer(inputSize, 'Name', 'input2', 'Normalization', 'none')

    convolution2dLayer([5 5], 4, 'Padding','same', 'Name','conv1_ch2')
    batchNormalizationLayer('Name','bn1_ch2')
    reluLayer('Name','relu1_ch2')
    maxPooling2dLayer([2 2], 'Stride', [2 2], 'Name','pool1_ch2')

    dropoutLayer(0.3, 'Name','dropout1_ch2')

    convolution2dLayer([3 3], 16, 'Padding','same', 'Name','conv2_ch2')
    batchNormalizationLayer('Name','bn2_ch2')
    reluLayer('Name','relu2_ch2')
    maxPooling2dLayer([2 2],'Stride', [2 2], 'Name','pool2_ch2')

    dropoutLayer(0.3, 'Name','dropout2_ch2')];

layers_ch3 = [
    imageInputLayer(inputSize, 'Name', 'input3', 'Normalization', 'none')

    convolution2dLayer([5 5], 4, 'Padding','same', 'Name','conv1_ch3')
    batchNormalizationLayer('Name','bn1_ch3')
    reluLayer('Name','relu1_ch3')
    maxPooling2dLayer([2 2], 'Stride', [2 2], 'Name','pool1_ch3')

    dropoutLayer(0.3, 'Name','dropout1_ch3')

    convolution2dLayer([3 3], 16, 'Padding','same', 'Name','conv2_ch3')
    batchNormalizationLayer('Name','bn2_ch3')
    reluLayer('Name','relu2_ch3')
    maxPooling2dLayer([2 2],'Stride', [2 2], 'Name','pool2_ch3')

    dropoutLayer(0.3, 'Name','dropout2_ch3')];

concat =  concatenationLayer(3, 3, 'Name','concat');

layers_lstm = [
   
    convolution2dLayer([1 1], 32, 'Name','merge_conv', 'Padding','same');
    functionLayer(@(x) permute(x, [2 3 1 4]))  
    functionLayer(@(x) dlarray(reshape(x, size(x,1),[], size(x,4)), 'TCB'), ...
        'Formattable', true, 'Name', 'reshapeForLSTM')
    lstmLayer(64, 'OutputMode','last', 'Name', 'lstm_stft')
                                   
    flattenLayer('Name','flatten')      
    dropoutLayer(0.3, 'Name','dropout3')
    fullyConnectedLayer(numClasses, 'Name','fc_final')
    softmaxLayer('Name','softmax')
    classificationLayer
];

lgraph = layerGraph();
lgraph = addLayers(lgraph, layers_ch1);
lgraph = addLayers(lgraph, layers_ch2);
lgraph = addLayers(lgraph, layers_ch3);

lgraph = addLayers(lgraph, concat);
lgraph = connectLayers(lgraph, 'dropout2_ch1', 'concat/in1');
lgraph = connectLayers(lgraph, 'dropout2_ch2', 'concat/in2');
lgraph = connectLayers(lgraph, 'dropout2_ch3', 'concat/in3');

lgraph = addLayers(lgraph, layers_lstm);
lgraph = connectLayers(lgraph, 'concat', 'merge_conv');


% lgraph = connectLayers(lgraph, 'concat', 'merge_conv');

sessions_idx = reshape(1:(sessions*trials*classes), [trials*classes, sessions])';
fold_acc = [];
fold_pred = [];
fold_true = [];

for i = 1:sessions
    test_range = sessions_idx(i, :);
    train_range = setdiff(1:sample_idx-1, test_range);

    % --- 검증 코드 삽입 (매 fold마다) ---
    disp(['Fold ', num2str(i), ...
        ' | train min/max: ', num2str(min(train_range)), ' ~ ', num2str(max(train_range)), ...
        ' | test min/max: ', num2str(min(test_range)), ' ~ ', num2str(max(test_range)), ...
        ' | train count: ', num2str(length(train_range)), ...
        ' | test count: ', num2str(length(test_range))]);
    % 혹은, 실제 겹침이 없는지도 체크
    if ~isempty(intersect(train_range, test_range))
        warning('Train/Test 인덱스에 중복이 있습니다!');
    end

    
    XTrain = X_stft(:,:,:,train_range);
    YTrain = categorical(y(train_range));
    XValid = X_stft(:,:,:,test_range);
    YValid = categorical(y(test_range));
    
    dsXTr1 = arrayDatastore(XTrain(:, :, 1, :), 'IterationDimension', 4);
    dsXTr2 = arrayDatastore(XTrain(:, :, 2, :), 'IterationDimension', 4);
    dsXTr3 = arrayDatastore(XTrain(:, :, 3, :), 'IterationDimension', 4);
    dsYTr = arrayDatastore(YTrain);
    dsTr = combine(dsXTr1,dsXTr2,dsXTr3,dsYTr);
    
    dsXVal1 = arrayDatastore(XValid(:, :, 1, :), 'IterationDimension', 4);
    dsXVal2 = arrayDatastore(XValid(:, :, 2, :), 'IterationDimension', 4);
    dsXVal3 = arrayDatastore(XValid(:, :, 3, :), 'IterationDimension', 4);
    dsYVal = arrayDatastore(YValid);
    dsVal = combine(dsXVal1,dsXVal2,dsXVal3,dsYVal);
    dsTest = combine(dsXVal1,dsXVal2,dsXVal3);
    
    % dsVal_env = transform(dsVal_env, @(data) {dlarray(data{1}, 'CBT')});
    options = trainingOptions('adam', ...
        'Shuffle','every-epoch', ...
        'GradientThreshold', 1.0, ...
        'L2Regularization', 1e-4, ...
        'ValidationPatience', 5, ...
        MaxEpochs=50, ...
        InitialLearnRate=5e-4, ...
        MiniBatchSize=64, ...
        ValidationData=dsVal, ...
        ValidationFrequency=10, ...
        OutputNetwork="best-validation", ...
        ExecutionEnvironment="auto", ...
        Plots="training-progress", ...
        Verbose=false);
   

    net = trainNetwork(dsTr, lgraph, options);
    YPred = classify(net, dsTest);
    acc = mean(YPred == YValid);
    fold_acc = [fold_acc; acc];
    fold_pred = [fold_pred; YPred];
    fold_true = [fold_true; YValid];
    fprintf('Fold %d 정확도: %.2f%%\n', i, acc*100);
end

figure;
confusionchart(fold_true, fold_pred, ...
    ColumnSummary="column-normalized", ...
    RowSummary="row-normalized");
title('Cross-validation Confusion Matrix');
fprintf('평균 CV 정확도: %.2f%%\n', mean(fold_acc)*100);

% 그냥 trial 데이터 하나에 대한 STFT 시각화 확인용
sel = 1;
figure;
for ch = 1:channels
    subplot(1,channels,ch);
    imagesc(X_stft(:,:,ch,sel));
    axis xy; colorbar; colormap jet;
    xlabel('Time Frame'); ylabel('Frequency Bin');
    title(sprintf('Sample %d, Channel %d', sel, ch));
end
sgtitle('STFT(dB, trial-normalized) Input Image');

% save the model
dsX1 = arrayDatastore(X_stft(:, :, 1, :), 'IterationDimension', 4);
dsX2 = arrayDatastore(X_stft(:, :, 2, :), 'IterationDimension', 4);
dsX3 = arrayDatastore(X_stft(:, :, 3, :), 'IterationDimension', 4);
dsY = arrayDatastore(categorical(y));
dsTrALL = combine(dsXTr1,dsXTr2,dsXTr3,dsYTr);
    
options_final = trainingOptions('adam', ...
    'Shuffle','every-epoch', ...
    'GradientThreshold', 1, ...
    'L2Regularization', 1e-4, ...
    MaxEpochs=30, ...
    InitialLearnRate=5e-4, ...
    MiniBatchSize=64, ... ...
    ExecutionEnvironment="auto", ...
    Plots="training-progress", ...
    Verbose=false);

net_final = trainNetwork(dsTrALL, lgraph, options_final);
save('model/trained_model_3ch_lstm.mat', 'net_final','inputSize');
fprintf("saved.");