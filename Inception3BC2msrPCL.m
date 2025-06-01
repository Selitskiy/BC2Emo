%% Clear everything 
clear all; close all; clc;

%ngpu = gpuDeviceCount();
%for i=1:ngpu
%    reset(gpuDevice(i));
%end

%AlexNet
%cdata = [0.2366 0.1427 0.1811 0.1603 0.1437 0.1804 0.1905 0.1327;... 
%    0.071 0.4026 0.1052 0.061 0.0817 0.1111 0.0552 0.056;...
%    0.1434 0.102 0.2648 0.1203 0.0823 0.1082 0.1494 0.1126;...
%    0.21 0.1979 0.2255 0.4939 0.2218 0.1941 0.1409 0.188;...
%    0.071 0.0317 0.0308 0.0222 0.1887 0.1407 0.0673 0.1019;...
%    0.0452 0.0363 0.0315 0.0122 0.0704 0.0664 0.0212 0.0208;...
%    0.1254 0.0363 0.0966 0.0566 0.0859 0.1111 0.2677 0.2009;...
%    0.0975 0.0506 0.0644 0.0744 0.1155 0.088 0.1076 0.1872];
%VGG16
%cdata = [0.2896 0.1148 0.1761 0.1518 0.1908 0.2309 0.1983 0.1686;...
%    0.0172 0.4894 0.0215 0.0093 0.0218 0.0498 0.012 0.0151;...
%    0.2746 0.1866 0.5483 0.2613 0.1908 0.1703 0.2365 0.1707;...
%    0.0373 0.0272 0.0587 0.04009 0.0155 0.0231 0.0319 0.0194;...
%    0.0767 0.0189 0.0172 0.02 0.2324 0.1024 0.0616 0.056;...
%    0.129 0.102 0.0766 0.0344 0.1775 0.2792 0.0921 0.1313;...
%    0.0523 0.0166 0.0365 0.0329 0.05 0.0685 0.1912 0.1334;...
%    0.1233 0.0446 0.0651 0.0895 0.1211 0.0758 0.1764 0.3056];
%GoogleNet
%cdata = [0.2344 0.1254 0.194 0.1346 0.157 0.1789 0.1473 0.1614;...
%    0.1075 0.6337 0.1403 0.1224 0.1507 0.1948 0.1069 0.0933;...
%    0.1606 0.0431 0.292 0.1052 0.081 0.0859 0.1331 0.1176;...
%    0.1204 0.0665 0.1532 0.4546 0.1078 0.0873 0.0977 0.0947;...
%    0.0394 0.003 0.0086 0.0165 0.1563 0.0729 0.034 0.0423;...
%    0.1118 0.0597 0.0802 0.0336 0.1359 0.2208 0.0907 0.0796;...
%    0.1262 0.0491 0.0916 0.0795 0.1049 0.0895 0.2507 0.1836;...
%    0.0996 0.0196 0.0401 0.0537 0.1063 0.07 0.1395 0.2274];
%ResNet
%cdata = [0.2724 0.1276 0.2362 0.224 0.2338 0.2417 0.2741 0.2174;...
%    0.0222 0.4887 0.0293 0.0143 0.031 0.0491 0.0205 0.0251;...
%    0.319 0.2349 0.4717 0.2398 0.2338 0.2908 0.2408 0.2489;...
%    0.0989 0.0642 0.1195 0.3923 0.0782 0.0606 0.0581 0.0681;...
%    0.0681 0.0076 0.0186 0.0193 0.1669 0.0758 0.0722 0.0516;...
%    0.0552 0.031 0.0208 0.0115 0.1155 0.1241 0.0524 0.0617;...
%    0.086 0.0242 0.0616 0.0335 0.0683 0.083 0.1877 0.137;...
%    0.0781 0.0219 0.0422 0.0651 0.0725 0.075 0.0942 0.1901];
%Inception
%cdata = [0.3577 0.1171 0.2047 0.1417 0.1472 0.2237 0.2068 0.1908;...
%    0.0115 0.5302 0.0093 0.0036 0.0197 0.0361 0.0106 0.0093;...
%    0.3047 0.2273 0.6521 0.219 0.2472 0.2821 0.2812 0.2138;...
%    0.0824 0.0347 0.0465 0.5433 0.0627 0.0426 0.0559 0.0653;...
%    0.0509 0.0008 0.005 0.0129 0.1922 0.0563 0.0234 0.043;...
%    0.0703 0.0597 0.0115 0.0064 0.1613 0.2172 0.0623 0.0516;...
%    0.0724 0.0098 0.0308 0.0136 0.0937 0.0722 0.2585 0.1736;...
%    0.0502 0.0204 0.0401 0.0594 0.0761 0.07 0.1013 0.2525];
%InceptRes
cdata = [0.2853 0.0702 0.1775 0.126 0.1556 0.1941 0.2309 0.1506;...
    0.0473 0.6579 0.0616 0.0537 0.0866 0.1241 0.0482 0.0653;...
    0.3018 0.1329 0.5705 0.2054 0.1951 0.2309 0.2351 0.1844;...
    0.1384 0.0725 0.1067 0.5276 0.1035 0.0729 0.0999 0.1234;...
    0.0423 0.0008 0.086 0.0093 0.1782 0.0786 0.0404 0.0516;...
    0.0502 0.0287 0.0072 0.0029 0.1444 0.1732 0.0418 0.0638;...
    0.0889 0.0287 0.0429 0.0279 0.0761 0.0743 0.2217 0.1578;...
    0.0459 0.0083 0.0251 0.0472 0.0606 0.0519 0.0822 0.203];

cdataf = flip(cdata,1);

xvalues = {'AN','CE','DS', 'HP', 'NE', 'SA', 'SC', 'SR'};
yvalues = {'AN','CE','DS', 'HP', 'NE', 'SA', 'SC', 'SR'};
yvaluesf = flip(yvalues);
h = heatmap(xvalues, yvaluesf, cdataf);%, 'FontSize', 15);

%ax = gca;
%ax.FontSize = 30;

%h.Title = 'T-Shirt Orders';
h.XLabel = 'Expression';
h.YLabel = 'Guess';

% Make heatmap fill the figure
%h.InnerPosition = [0.1 0.1 0.8 0.8];

%h.OuterPosition = [0 0 0.85 1]; % Set width and height equal

%%

Line1X = [1  2  4  8  16];
Line1Yc = [0.6221 0.6046 0.6784 0.6567 0.6668];
Line1Ys = [0.7514 0.7744 0.7624 0.7344 0.7437];
            plot(Line1X, Line1Yc, Line1X, Line1Ys);
            %bar(ActTS(k,1:nClasses*nModels));
            %stairs(ActTS(k,1:nClasses*nModels));
            ax = gca;
            ax.FontSize = 40;
            xlabel('Ensemble size', 'FontSize', 40);
            ylabel('Accuracy', 'FontSize', 40);

            %%
Line1X = [1 2 4 6 8];
Line1Yc = [0.2810 0.3009 0.3256 0.3450 0.3494];
Line1Ys = [0.7207 0.6967 0.7018 0.6788 0.6770];

            plot(Line1X, Line1Yc, Line1X, Line1Ys);
            %bar(ActTS(k,1:nClasses*nModels));
            %stairs(ActTS(k,1:nClasses*nModels));
            ax = gca;
            ax.FontSize = 40;
            xlabel('Ensemble size', 'FontSize', 40);
            ylabel('Accuracy', 'FontSize', 40);

%% Dataset root folder template and suffix
dataFolderTmpl = '~/data/BC2_Sfx';
dataFolderSfx = '1072x712';

kfold_pref = "";


% Create imageDataset of all images in selected baseline folders (5 possible folds)
[baseSet, dataSetFolder] = createBCbaselineIDS6b1(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
%[baseSet, dataSetFolder] = createBCbaselineIDS6b2(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
%[baseSet, dataSetFolder] = createBCbaselineIDS6b3(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
%[baseSet, dataSetFolder] = createBCbaselineIDS6b4(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
%[baseSet, dataSetFolder] = createBCbaselineIDS6b5(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);

trainingSet = baseSet;

% Count number of the classes ('stable' - presrvation of the order - to use
% later for building confusion matrix)
labels = unique(trainingSet.Labels, 'stable');
[nClasses, ~] = size(labels);

% Print image count for each label
%trainCountTable = countEachLabel(trainingSet);
%trainCount = sum(table2array(trainCountTable(:,2)), 1);
[trainCount, ~] = size(trainingSet.Files);

t1 = clock();
                        
%% Split Database into Training & Test Sets in the ratio 80% to 20% (usually comment out)
%[trainingSet, testSet] = splitEachLabel(baseSet, 0.4, 'randomize'); 

%% Swarm of models
nModels = 7;
myNets = [];
save_net_fileT = '~/data/in_swarm';
%save_s1net_fileT = '~/data/in_swarm1_sv';
save_s2net_fileT = '~/data/in_swarm2CL_sv';

mb_size = 128;

for s=1:nModels
    n_ll = 315;
    % Load saved model if exists
    save_net_file = strcat(save_net_fileT, int2str(s), kfold_pref, '.mat');
    if isfile(save_net_file)
        load(save_net_file, 'myNet');
    end
       
    if exist('myNet') == 0
        
        cNet = Incept3Net(nClasses, 0.01, 30, mb_size);
        cNet = cNet.Create();

        % Train the Network 
        % This process usually takes about 30 minutes on a desktop GPU. 
    
        % Shuffle training set for more randomness
        trainingSetS = shuffle(trainingSet);
    
        while (exist('TInfo')==0) || (TAcc < 90.)

            % GPU on
            gpuDevice(1);
            reset(gpuDevice(1));

            [cNet, TInfo] = cNet.Train(trainingSetS);

            % GPU off
            delete(gcp('nocreate'));
            gpuDevice([]);

            %[~, TAccLast] = size(TInfo.TrainingAccuracy);
            TAcc = TInfo.TrainingAccuracy(end);
        end
        clear('TInfo');
        
        myNet = cNet.trainedNet;
        save(save_net_file, 'myNet');

    end
    
    myNets = [myNets, myNet];
    
    clear('myNet');
    clear('cNet');
    clear('trainingSetS');
    
end

%Mem2 = zeros([ngpu 1]);
%OMem2 = 0;
%for i=1:ngpu
%    dev = gpuDevice(i);
%    Mem2(i) = dev.TotalMemory - dev.AvailableMemory;
%    OMem2 = OMem2 + Mem2(i);
%end
t2 = clock();
fprintf('Incept3 training N images:%d time:%.3f, models %d\n', trainCount, etime(t2, t1), nModels);


%% Mem cleanup
%for i=1:ngpu
%    reset(gpuDevice(i));
%end

            % GPU on
            gpuDevice(1);
            reset(gpuDevice(1));
            % GPU off
            delete(gcp('nocreate'));
            gpuDevice([]);


%% Traditional accuracy (usually comment out) DEBUG
%predictedLabels = classify(myNet, testSet); 
%accuracy = mean(predictedLabels == testSet.Labels)

%predictedScores = predict(myNet, testSet);
%[nImages, ~] = size(predictedScores);
%for k=1:nImages
%    maxScore = 0;
%    maxScoreNum = 0;
%    maxScoreClass = "S";
%    correctClass = testSet.Labels(k);
%    for l=1:nClasses
%        if maxScore <= predictedScores(k, l)
%            maxScore = predictedScores(k, l);
%            maxScoreNum = l;
%            maxScoreClass = myNet.Layers(25).Classes(l);
%        end
%    end   
%    fprintf("%s %f %s \n", correctClass, maxScore, maxScoreClass);
%end



%% Reliability training datasets
% Create imageDataset vector of images in selected makeup folders (5 possible folds)
[testRSets, testRDataSetFolders] = createBCtestIDSvect6b1(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
%[testRSets, testRDataSetFolders] = createBCtestIDSvect6b2(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
%[testRSets, testRDataSetFolders] = createBCtestIDSvect6b3(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
%[testRSets, testRDataSetFolders] = createBCtestIDSvect6b4(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
%[testRSets, testRDataSetFolders] = createBCtestIDSvect6b5(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);

t1 = clock();

%% Create Matrix of Softmax Activations
[nMakeups, ~] = size(testRSets);
nImgsTot = 0;

for i=1:nMakeups
    [nImages, ~] = size(testRSets{i}.Files);
    nImgsTot = nImgsTot + nImages;
end

Act = zeros([nImgsTot nClasses nModels]);
Verd = zeros([nImgsTot nModels]);
Strong = zeros([nImgsTot nModels]);

Act2 = zeros([nImgsTot nModels]);

dimLabel = 1;
ActS = zeros([nImgsTot nClasses*nModels+dimLabel]);

nEnsLabels = nModels + 1;
% regression dimension (same as dimLabel here)
nRealCLOut = dimLabel;
% threwshold dimansion
nTrOut = 1;
% size of the last predictions table (state to use in loss calculation) 
nMem = 8192;
% technical output between state layere and loss
nCLOut = nRealCLOut + nTrOut + 2*nMem;

VerdS = zeros([nImgsTot nCLOut]);

mb_sizeS = 64;


%% Populate Matrix of Softmax Activations
nImgsCur = 1;
for i=1:nMakeups   
    [nImages, ~] = size(testRSets{i}.Files);
    
    fprintf('Makeup # %d/%d\n', i, nMakeups);    
            
    %% Walk through model Swarm
    ActPF = zeros([nImages nClasses nModels]);
    VerdPF = zeros([nImages nModels]);
    testRSet = testRSets{i};
    %par
    for s=1:nModels 
        
        predictedLabels = classify(myNets(s), testRSet);
        predictedScores = predict(myNets(s), testRSet);
        ActPF(:, :, s) = predictedScores;
        VerdPF(:, s) = (testRSet.Labels == predictedLabels);

    end
    Act(nImgsCur:nImgsCur + nImages - 1, :, :) = ActPF(:, :, :);        
    Verd(nImgsCur:nImgsCur + nImages - 1, :) = VerdPF(:, :);
    
    nImgsCur = nImgsCur + nImages;
    
end

%% Build Uncertainty Shape Descriptor 
[ActS, VerdS] = makeUSDstrong(Act, Verd, ActS, VerdS, nClasses, 0, nImgsTot, nModels, dimLabel);


%for i=1:ngpu
%    reset(gpuDevice(i));
%end

            % GPU on
            gpuDevice(1);
            reset(gpuDevice(1));
            % GPU off
            delete(gcp('nocreate'));
            gpuDevice([]);

%% Train Supervisor model
save_s2net_file = strcat(save_s2net_fileT, int2str(nModels), kfold_pref, '.mat');
%if isfile(save_s2net_file)
%    load(save_s2net_file, 'super2Net');
%else
%    clear('super2Net');
%end

Yt = categorical(VerdS');
[nDim, nVerdicts] =  size(countcats(Yt));

%if exist('super2Net') == 0

    nLayer1 = nClasses*nModels+1+dimLabel;
    nLayer2 = 2*nClasses*nModels+1+dimLabel;
    nLayer3 = 2*nClasses*nModels+1+dimLabel;


        sOptions = trainingOptions('adam', ...
        'ExecutionEnvironment','auto',...
        'Shuffle', 'every-epoch',...
        'MiniBatchSize', mb_sizeS, ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',200, ...
        'Verbose',true, ...
        'Plots','training-progress');

        %'LearnRateSchedule', 'piecewise',...
        %'LearnRateDropPeriod', 5,...
        %'LearnRateDropFactor', 0.9,...

    %sup_name = 'relu';
    %sup_name = 'tanh';
    %sup_name = 'sig';
    %sup_name = 'gelu';
    %sup_name = 'lrrelu';
    sup_name = 'crelu';

    sLayers = [
        featureInputLayer(nClasses*nModels+dimLabel)
        %fullyConnectedM1ReluLayer("L1", nClasses*nModels+dimLabel, dimLabel, nLayer1)
        %fullyConnectedM1ReluLayer("L2", nLayer1, dimLabel, nLayer2)
        %fullyConnectedM1TanhLayer("L1", nClasses*nModels+dimLabel, dimLabel, nLayer1)
        %fullyConnectedM1TanhLayer("L2", nLayer1, dimLabel, nLayer2)
        %fullyConnectedM1SigLayer("L1", nClasses*nModels+dimLabel, dimLabel, nLayer1)
        %fullyConnectedM1SigLayer("L2", nLayer1, dimLabel, nLayer2)
        %fullyConnectedM1GeluLayer("L1", nClasses*nModels+dimLabel, dimLabel, nLayer1)
        %fullyConnectedM1GeluLayer("L2", nLayer1, dimLabel, nLayer2)
        %fullyConnectedM1LrReluLayer("L1", nClasses*nModels+dimLabel, dimLabel, nLayer1, 1)
        %fullyConnectedM1LrReluLayer("L2", nLayer1, dimLabel, nLayer2, 1)
        fullyConnectedM1CreluLayer("L1", nClasses*nModels+dimLabel, dimLabel, nLayer1)
        fullyConnectedM1CreluLayer("L2", nLayer1, dimLabel, nLayer2)
        fullyConnectedCLLayer("L3", nLayer2, dimLabel, nRealCLOut, nCLOut, nEnsLabels, nMem)
        TCLmRegression("L4", dimLabel, nMem)
    ];
    sgraph = layerGraph(sLayers);
    

                % GPU on
            gpuDevice(1);
            reset(gpuDevice(1));

    super2Net = trainNetwork(ActS, VerdS, sgraph, sOptions);


    % Verify
    %supervisorPredictedScorest = predict(super2Nett, ActSt);
    %rmset = sqrt( sum(( (supervisorPredictedScorest - VerdSt) .^ 2), 'all') );
    supervisorPredictedScores = predict(super2Net, ActS);
    rmse = sqrt( sum(( (supervisorPredictedScores - VerdS) .^ 2), 'all') );
    TrTrain = supervisorPredictedScores(1, 2);
    %rmsv = sqrt( sum(( (supervisorPredictedScores - supervisorPredictedScorest) .^ 2), 'all') );

    
    %save(save_s2net_file, 'super2Net');


            % GPU off
            %delete(gcp('nocreate'));
            %gpuDevice([]);

%end

t2 = clock();
fprintf('Incept3 Supervisor training N images:%d time:%.3f, models %d\n', nImgsTot, etime(t2, t1), nModels);

%% Continous and Active and MCMC-like Active flags
contF = 0;
c_pref = '';
if contF
    c_pref = strcat(c_pref, 'c');
end

structF = 1;
if structF
    c_pref = strcat(c_pref, 's');
end

nOrcaleLimit = 0; %0.001;
if nOrcaleLimit > 0
    c_pref = strcat(c_pref, num2str(nOrcaleLimit), 'a');
end

mcmcF = 0;
if mcmcF
    c_pref = strcat(c_pref, 'm');
end

%% Makeup datasets
mkDataSetFolder = strings(0);
mkLabel = strings(0);

% Create imageDataset vector of images in selected makeup folders
if structF
    [testSets, testDataSetFolders] = createBCtestIDSvect6b(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
else
    [testSets, testDataSetFolders] = createBCtestIDSvect6bWhole(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
end

%%
[nMakeups, ~] = size(testSets);

mkTable = cell(nMakeups, nClasses+4);




%% Write per-image scores to a file
fd = fopen( strcat('predict_in_6bmsrTr',int2str(nModels), kfold_pref, c_pref, '.', sup_name, '.txt'),'w' );

fprintf(fd, "CorrectClass MeanPredictScore PredictClassMax PredictClassTr TrustScore TrustThresholdTr TrustThresholdCurr TrustFlag05 TrustFlagTr TrustFlagCurr TrustScoreLb FileName");
for l=1:nClasses
    fprintf(fd, " %s", myNets(1).Layers(n_ll).Classes(l));
end
fprintf(fd, "\n");


%%
aOpts = trainingOptions('adam',...
    'ExecutionEnvironment','auto',...
    'InitialLearnRate', 0.001,...
    'LearnRateSchedule', 'piecewise',...
    'LearnRateDropPeriod', 5,...
    'LearnRateDropFactor', 0.9,...
    'MiniBatchSize', mb_size,...
    'MaxEpochs', 5);

% Active re-training
[nTrainImg,~] = size(trainingSet.Files);
activeI = randperm(nTrainImg, mb_size);
activeFiles = trainingSet.Files(activeI);
activeLabels = trainingSet.Labels(activeI);
activeSet = imageDatastore(activeFiles); 
activeSet.Labels = activeLabels;
activeSet.ReadFcn = trainingSet.ReadFcn;

% Continous re-training
contI = randperm(nImgsTot, mb_sizeS);
contActTS = zeros([mb_sizeS nClasses*nModels+dimLabel]);
contVerdTS = zeros([mb_sizeS nCLOut]);
contActTS = ActS(contI, :);
contVerdTS = VerdS(contI, :);


%% Create Matrix of Softmax Activations
[nMakeups, ~] = size(testSets);

nImgsTotT = 0;

for i=1:nMakeups
    [nImages, ~] = size(testSets{i}.Files);
    nImgsTotT = nImgsTotT + nImages;
end

fprintf('Incept3 test N images:%d, models %d\n', nImgsTotT, nModels);

ActT = zeros([nImgsTotT nClasses nModels]);
VerdT = zeros([nImgsTotT nModels]);
StrongT = zeros([nImgsTotT nModels]);
ActTS = zeros([nImgsTotT nClasses*nModels+dimLabel]);
VerdTS = zeros([nImgsTotT nCLOut]);


% Supervisor network
supervisorPredictedScores = zeros([nImgsTotT nCLOut]);
cOptions = trainingOptions('adam', ...
    'ExecutionEnvironment','auto',...
    'Shuffle', 'every-epoch',...
    'MiniBatchSize', mb_sizeS, ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',10, ...
    'Verbose',true);%, ...
    %'Plots','training-progress');

%% Populate Matrix of Softmax Activations
predictedLabelsSwarm = cell(nMakeups, nModels);
predictedScoresSwarm = cell(nMakeups, nModels);
nImgsCur = 1;

activeCurI = 1;
nOracleAnsw = 0;
difPredLabelMax = 0;

contCurI = 1;


for i=1:nMakeups   
    [nImages, ~] = size(testSets{i}.Files);
    
    fprintf('Makeup # %d/%d\n', i, nMakeups);
        
    ActTPF = zeros([nImages nClasses nModels]);
    VerdTPF = zeros([nImages nModels]);
    testSet = testSets{i};

                % GPU on
            %gpuDevice(1);
            %reset(gpuDevice(1));
    
    for j=1:nImages
        fprintf('Image # %d/%d of Makeup %d/%d\n', j, nImages, i, nMakeups);
        img = readimage(testSet, j);
        testLabels = testSet.Labels(j);



        % Walk through model Swarm
        %par
        for s=1:nModels 
            % Test main network performance
            predictedLabels = classify(myNets(s), img);
            predictedLabelsSwarm{i, s} = predictedLabels;
            predictedScores = predict(myNets(s), img);
            predictedScoresSwarm{i, s} = predictedScores;
        
            ActTPF(j, :, s) = predictedScores;        
            VerdTPF(j, s) = (testLabels == predictedLabels);
        end
        k = nImgsCur + j - 1;
        ActT(k, :, :) = ActTPF(j, :, :);
        VerdT(k, :) = VerdTPF(j, :);

        % Make Uncertainty Shape Descriptor
        [ActTS, VerdTS] = makeUSDstrong(ActT, VerdT, ActTS, VerdTS, nClasses, k, 0, nModels, dimLabel);
        supervisorPredictedScores(k,:) = predict(super2Net, ActTS(k, :));

        % Uncertainty Descriptor visualization
        if ActTS(k,end) == 7
            bar(ActTS(k,1:nClasses*nModels));
            %stairs(ActTS(k,1:nClasses*nModels));
            ax = gca;
            ax.FontSize = 40;
            xlabel('Sorted activations', 'FontSize', 40);
            ylabel('Activation value', 'FontSize', 40);
        end
        %

        % Find label with number of occurance as predicted by superNet
        % Ensemble voting
        for s=1:nModels 
            predictedScoresS(:, :, s) = predictedScoresSwarm{i, s};
            predictedLabelsS(:, s) = predictedLabelsSwarm{i, s};
        end

        [predictedLabels, predictedScores, predictedLabelsCat, predictedLabelSuper, predictedLabelScoreSuper] = ensemblePredictedLabels(...
            predictedLabelsS, predictedScoresS, supervisorPredictedScores(k,:), 1, nClasses);

        %nCatMatches = countcats(predictedLabelsS, 2);
        %[MaxCountLabels, MI] = max(nCatMatches, [], 2);
        %predictedLabelsCat = (categories(predictedLabelsS));
        %predictedLabels = predictedLabelsCat(MI);
    
        %predictedScores = mean(predictedScoresS, 3);

        %dCatMatches = abs(nCatMatches(1,:) - round(supervisorPredictedScores(k, 1)));
        %[minDistVotes, DI] = sort(dCatMatches, 2, 'ascend');

        %predictedLabelScores = zeros([nModels 1]);
        %dMin = minDistVotes(1);
        %for l=1:nClasses
        %    if minDistVotes(l) > dMin
        %        break;
        %    end
        %    predictedLabelScores(l) = mean(predictedScoresS(1, DI(l), predictedLabelsS(1,:) == predictedLabelsCat(DI(l))));
        %end
        %[predictedLabelScoreSuper, PI] = max(predictedLabelScores, [], 1);
        %predictedLabelSuper = predictedLabelsCat(DI(PI));


        correctClass = testSets{i}.Labels(j);
        predictedLabel = predictedLabels{1};

        idxsPredictedCat = find(predictedLabelsCat == categorical(predictedLabels(1)));
        idxsWonModels = predictedLabelsS(1,:) == predictedLabels(1,:);
        predictedLabelScore = mean(predictedScoresS(1, idxsPredictedCat, idxsWonModels));

        fprintf(fd, "%s %f %s %s %f %f %f %d %d %d %f %s", correctClass, predictedLabelScore, predictedLabel, predictedLabelSuper{1},...
            supervisorPredictedScores(k, 1),...
            TrTrain,...
            supervisorPredictedScores(k, 2),...
            supervisorPredictedScores(k, 1) > (nEnsLabels-1)/2.,...
            supervisorPredictedScores(k, 1) > TrTrain,...
            supervisorPredictedScores(k, 1) > supervisorPredictedScores(k, 2),...
            VerdTS(k, 1),...
            testSets{i}.Files{j});
        for l=1:nClasses
            fprintf(fd, " %f", predictedScores(1, l));
        end
        fprintf(fd, "\n");



        % Ask Oracle if in doubt (prediction trust score is below threshold
        % trust score)
        difPredLabel = supervisorPredictedScores(k, 2) - supervisorPredictedScores(k, 1);
        if (((mcmcF == 0) && (supervisorPredictedScores(k, 1) < supervisorPredictedScores(k, 2)) && (nOracleAnsw/k < nOrcaleLimit)) ||...
                ((mcmcF == 1) && (difPredLabel > difPredLabelMax)))... 
                

            % Active re-training - insert current test image
            activeFiles = activeSet.Files;
            activeLabels = activeSet.Labels;
            activeFiles(activeCurI) = testSet.Files(j);
            activeSet = imageDatastore(activeFiles);
            activeSet.Labels = activeLabels;
            activeSet.Labels(activeCurI) = testSet.Labels(j);
            activeSet.ReadFcn = trainingSet.ReadFcn;

            % GPU on
            %gpuDevice(1);
            %reset(gpuDevice(1));

            for s=1:nModels 
                [myNets(s), TInfo] = trainNetwork(activeSet, myNets(s).layerGraph, aOpts);
            end

                        % GPU off
            %delete(gcp('nocreate'));
            %gpuDevice([]);

            activeCurI = 1 + mod(nOracleAnsw, mb_size);
            %if activeCurI == 0
            %    activeCurI = mb_size;
            %end

            nOracleAnsw = nOracleAnsw + 1;
            difPredLabelMax = difPredLabel;
        end


        % Naive continous learning, retrain every prediction
        if contF
            %sLayers = super2Net.Layers;
            %sgraph = layerGraph(sLayers);
            contActTS(contCurI, :) = ActTS(k, :);
            contVerdTS(contCurI, :) = VerdTS(k, :);

            % GPU on
            %gpuDevice(1);
            %reset(gpuDevice(1));

            super2Net = trainNetwork(contActTS, contVerdTS, super2Net.layerGraph, cOptions);

                        % GPU off
            %delete(gcp('nocreate'));
            %gpuDevice([]);

            contCurI = 1 + mod(k, mb_sizeS);
        end
    
    end
    nImgsCur = nImgsCur + nImages;

                % GPU off
            %delete(gcp('nocreate'));
            %gpuDevice([]);
    
end
                % GPU off
            delete(gcp('nocreate'));
            gpuDevice([]);

fclose(fd);
