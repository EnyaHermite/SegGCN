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
psicnnDir = '/media/huanlei/Data/PycharmProjects/SPH3D-GCN';
dataFolder = 'scannet-3cm-0.5';
resultFolder = sprintf('results_augment_50_%s',dataFolder);
indexFolder = sprintf('block_index_%s',dataFolder);
fullDir = '/media/huanlei/Data/Datasets/ScanNet';
voxelDir = '/media/huanlei/Data/Datasets/ScanNet-3cm';
test_folder = 'train';
debug_index = false;

compare_num = 2;

total_intersect = zeros(num_cls,compare_num);
total_union = zeros(num_cls,compare_num);
total_seen = zeros(num_cls,compare_num);
merged_correct = zeros(1,compare_num);
merged_seen = zeros(1,compare_num);

scene_names = textread(fullfile(voxelDir,'scannetv2_val.txt'),'%s');
for i = 1:numel(scene_names)
    scene = scene_names{i};
    voxelCloud = load(fullfile(voxelDir,test_folder,strcat(scene,'.txt')));
        
    gt_label = voxelCloud(:,end);
    gt_label_40 = labelid_set(gt_label+1);
    predictions = zeros(numel(gt_label),numel(classes));
    
    %% merge the predictions
    pred_files = dir(fullfile(psicnnDir,'log_scannet',resultFolder,sprintf('%s_*.mat',scene)));
    if isempty(pred_files)
        error('scene not found');
    end
    for k = 1:numel(pred_files)
        load(fullfile(pred_files(k).folder,pred_files(k).name));
        load(fullfile(strrep(pred_files(k).folder,resultFolder,indexFolder),pred_files(k).name));

        in_index = data(:,8)==1;
        inner_pt = data(in_index,1:3);
        pred_logits = data(in_index,9:end);
        pred_logits = pred_logits./sqrt(sum(pred_logits.^2,2)); % normlize to unit vector
        pred_logits = exp(pred_logits)./sum(exp(pred_logits),2); % further normlize to probability/confidence

        block2full_index = index(in_index)+1;

        predictions(block2full_index,:) = predictions(block2full_index,:) + pred_logits;
    end
    [~,pred_label] = max(predictions,[],2);
    pred_label = pred_label - 1; 
    pred_label_40 = labelid_set(pred_label+1);
    
    a = 1;
    for l = 1:numel(classes)
        total_intersect(l,a) = total_intersect(l,a) + sum((pred_label==(l-1)) & (gt_label==(l-1)));
        total_union(l,a)  = total_union(l,a) + sum((pred_label==(l-1)) | (gt_label==(l-1)));
        total_seen(l,a) = total_seen(l,a) + sum((gt_label==(l-1)));
    end
    merged_correct(a) = sum(total_intersect(2:end,a));
    merged_seen(a) = sum(total_seen(2:end,a));
    for k = 1:numel(pred_label_40)
        gt_id = gt_label_40(k);
        pred_id = pred_label_40(k);
        confusion_voxel(gt_id,pred_id) = confusion_voxel(gt_id,pred_id)+1;
    end
    
    % assign neighbor to the full point cloud based on the nearest neighbor
    % in the voxelized point cloud
    a = 2; 
    fullCloud = load(fullfile(fullDir,test_folder,strcat(scene,'.txt')));
    [IDX, D] = knnsearch(voxelCloud(:,1:3),fullCloud(:,1:3));
    gt_label = fullCloud(:,end);
    pred_label = pred_label(IDX(:)); % pred_label in the original full point cloud   
    pred_label = mode(pred_label,2);
    for l = 1:numel(classes)
        total_intersect(l,a) = total_intersect(l,a) + sum((pred_label==(l-1)) & (gt_label==(l-1)));
        total_union(l,a)  = total_union(l,a) + sum((pred_label==(l-1)) | (gt_label==(l-1)));
        total_seen(l,a) = total_seen(l,a) + sum((gt_label==(l-1)));
    end
    merged_correct(a) = sum(total_intersect(2:end,a));
    merged_seen(a) = sum(total_seen(2:end,a));
    
    gt_label_40 = labelid_set(gt_label+1);
    pred_label_40 = labelid_set(pred_label+1);
    for k = 1:numel(pred_label_40)
        gt_id = gt_label_40(k);
        pred_id = pred_label_40(k);
        confusion_full(gt_id,pred_id) = confusion_full(gt_id,pred_id)+1;
    end

    fprintf('%s: %.2f%%, %.2f%%\n',scene,100*merged_correct(1)./(merged_seen(1)+eps),100*merged_correct(2)./(merged_seen(2)+eps));
end

%% metric evaluation on the 20 valid label ids

OA = merged_correct./(merged_seen+eps);
% OA_ = sum(total_intersect(2:end,:))./sum(total_seen(2:end,:));
class_iou = total_intersect./(total_union+eps);
class_acc = total_intersect./(total_seen+eps);
fprintf('==================================class_OA==================================\n')
disp(OA(:)');
fprintf('=====================================end=====================================\n')
fprintf('==================================class_iou==================================\n')
disp(class_iou');
fprintf('=====================================end=====================================\n')
fprintf('==================================class_acc==================================\n')

%         figure(1);clf;visualize(fullCloud(:,1:3),pred_label),title('SPH3D-GCN results')
%         figure(2);clf;visualize(fullCloud(:,1:3),gt_label),title('ground truth')
disp(class_acc');
fprintf('=====================================end=====================================\n');
disp([mean(class_iou(2:end,:));mean(class_acc(2:end,:))]);


IoU_1 = getiou(labelid_set(2:end),confusion_voxel);
IoU_2 = getiou(labelid_set(2:end),confusion_full);
% save(fullfile(psicnnDir, 's3dis_seg', sprintf('%s_metric',AreaID)),'merged_correct','merged_seen','total_intersect','total_union','total_seen'); 