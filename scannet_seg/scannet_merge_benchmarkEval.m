clc;clear;close all;

classes = {'other20','wall','floor','cabinet','bed','chair',...
           'sofa','table','door','window','bookshelf',...
           'picture','counter','desk','curtain','refridgerator',...
           'shower curtain','toilet','sink','bathtub','otherfurniture'};
num_cls = length(classes);
labelid_set = [40 1:12 14 16 24 28 33 34 36 39]; % 0 to 40
confusion_voxel = zeros(40,40);
confusion_full = zeros(40,40);
       
% compute evaluation metric by removing overlapping between blocks
baseDir = pwd;
baseDir(baseDir=='\') == '/';
str = split(pwd,'/');
sph3dgcnDir = join(str(1:end-1),'/');
sph3dgcnDir = sph3dgcnDir{:};

dataFolder = 'scannet-3cm';
resultFolder = 'results-noAug-60';
indexFolder = sprintf('block_index',dataFolder);
fullDir = '/media/huanlei/Data/Datasets/ScanNet';
voxelDir = '/media/huanlei/Data/Datasets/ScanNet-3cm';
test_folder = 'train';

compare_num = 2;

scene_names = textread(fullfile(voxelDir,'scannetv2_val.txt'),'%s');
for i = 1:numel(scene_names)
    scene = scene_names{i};
    voxelCloud = load(fullfile(voxelDir,test_folder,strcat(scene,'.txt')));
        
    gt_label = voxelCloud(:,end);
    gt_label_40 = labelid_set(gt_label+1);
    predictions = zeros(numel(gt_label),numel(classes));
    
    %% merge the predictions
    pred_files = dir(fullfile(sph3dgcnDir,'log_scannet',resultFolder,sprintf('%s_*.mat',scene)));
    index_files = dir(fullfile(sph3dgcnDir,'log_scannet',indexFolder,sprintf('%s_*.mat',scene)));
    if isempty(pred_files)
        error('scene not found');
    end
    for k = 1:numel(pred_files)
        load(fullfile(pred_files(k).folder,pred_files(k).name));
        load(fullfile(index_files(k).folder,index_files(k).name));

        in_index = data(:,8)==1;
        inner_pt = data(in_index,1:3);
        pred_logits = data(in_index,9:end);
        pred_logits = pred_logits./sqrt(sum(pred_logits.^2,2)); % normlize to unit vector
        pred_logits = exp(pred_logits)./sum(exp(pred_logits),2); % further normlize to probability/confidence

%         index = index(:);
        block2full_index = index(in_index)+1;

        predictions(block2full_index,:) = predictions(block2full_index,:) + pred_logits;
    end
    [~,pred_label] = max(predictions,[],2);
    pred_label = pred_label - 1; 
    pred_label_40 = labelid_set(pred_label+1);
    
    for k = 1:numel(pred_label_40)
        gt_id = gt_label_40(k);
        pred_id = pred_label_40(k);
        confusion_voxel(gt_id,pred_id) = confusion_voxel(gt_id,pred_id)+1;
    end
    
    % assign neighbor to the full point cloud based on the nearest neighbor
    % in the voxelized point cloud
    fullCloud = load(fullfile(fullDir,test_folder,strcat(scene,'.txt')));
    [IDX, D] = knnsearch(voxelCloud(:,1:3),fullCloud(:,1:3));
    gt_label = fullCloud(:,end);
    gt_label_40 = labelid_set(gt_label+1);
    pred_label_40 = pred_label_40(IDX(:)); % pred_label in the original full point cloud   
    for k = 1:numel(pred_label_40)
        gt_id = gt_label_40(k);
        pred_id = pred_label_40(k);
        confusion_full(gt_id,pred_id) = confusion_full(gt_id,pred_id)+1;
    end
end

%% metric evaluation on the 20 valid label ids
IoU_1 = getiou(labelid_set(2:end),confusion_voxel);
IoU_2 = getiou(labelid_set(2:end),confusion_full);
fprintf('%15s: voxel(%.2f%%), full(%.2f%%)\n','mIoU',mean(IoU_1)*100,mean(IoU_2)*100); 
for i = 1:numel(classes)-1
   fprintf('%15s: voxel(%.2f%%), full(%.2f%%)\n',classes{i+1},IoU_1(i)*100,IoU_2(i)*100); 
end
% save(fullfile(psicnnDir, 's3dis_seg', sprintf('%s_metric',AreaID)),'merged_correct','merged_seen','total_intersect','total_union','total_seen'); 