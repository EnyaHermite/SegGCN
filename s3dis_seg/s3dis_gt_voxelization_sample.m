clc;clear;close all;

voxel_size = 1; % unit: centimeter


readDir = '/media/huanlei/Data/Datasets/S3DIS-Aligned';
writeDir = strcat(readDir,sprintf('-%dcm',voxel_size));

Areas = {'Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6'};

roomSize = cell(numel(Areas),1);
for i = 5%1:numel(Areas)
    Builds = dir(fullfile(readDir, Areas{i}));
    Builds = Builds(3:end);
    dirFlags = [Builds.isdir];
    Builds = Builds(dirFlags); % Extract only those that are directories
    
    roomSize{i} = [];
    for j = 1:numel(Builds)
        objects = dir(fullfile(Builds(j).folder, Builds(j).name, 'Annotations', '*.txt'));
        
        count = 0;
        for k = 1:numel(objects)
            readpath = fullfile(readDir, Areas{i}, Builds(j).name, 'Annotations',  objects(k).name);
            writepath = fullfile(writeDir, Areas{i}, Builds(j).name, 'Annotations',  objects(k).name);
            
            if ~exist(fullfile(writeDir, Areas{i}, Builds(j).name, 'Annotations'))
                    mkdir(fullfile(writeDir, Areas{i}, Builds(j).name, 'Annotations'));
            end
            
            pt = load(readpath);
            Cloud = pointCloud(pt(:,1:3),'color',uint8(pt(:,4:6)));
            sampleCloud = pcdownsample(Cloud,'gridAverage', voxel_size/100);
            new_pt = [double(sampleCloud.Location) double(sampleCloud.Color)];
            dlmwrite(writepath, new_pt,'delimiter',' ');
            
            count = count + size(new_pt,1);
        end
        roomSize{i} = [roomSize{i} count];
    end
    
end
