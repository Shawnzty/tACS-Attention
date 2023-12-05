% function Electrode_pos(varargin)
% Electrode_pos(): Open GUI to selsct CSV file
% Electrode_pos(filepath): Use filepath

% Colorset
colors = [0, 0.4470, 0.7410;
    0.8500, 0.3250, 0.0980;
    0.9290, 0.6940, 0.1250;
    0.4940, 0.1840, 0.5560;
    0.4660, 0.6740, 0.1880;
    0.3010, 0.7450, 0.9330;
    0.6350, 0.0780, 0.1840];

% Import CSV file
[file_pos,path_pos] = uigetfile('*.csv');
sens_pos = readtable(fullfile(path_pos,file_pos));

% Convert
theta = table2array(sens_pos(:,2));
phi = table2array(sens_pos(:,3));
type = table2array(sens_pos(:,4));
labels = table2array(sens_pos(:,1));
color = zeros(length(type),3);
color(type == 1,:) = repmat([.8 .8 .8],[sum(type == 1),1]);
for idx = 1:7
    color(type == idx+1,:) = repmat(colors(idx,:),[sum(type == idx+1),1]);
end
display = find(type ~= 0);
radius = theta/90;
x = radius.*cos(pi*phi/180);
y = -radius.*sin(pi*phi/180);

% Drawing
figure('visible','off');
hold on
rectangle('Position',[-5/4 -5/4 5/2 5/2],'Curvature',[1,1],'EdgeColor',[.6 .6 .6],'LineStyle','-');
rectangle('Position',[-1 -1 2 2],'Curvature',[1,1],'EdgeColor',[.2 .2 .2],'LineStyle','-');
rectangle('Position',[-3/4 -3/4 3/2 3/2],'Curvature',[1,1],'EdgeColor',[.6 .6 .6],'LineStyle','-');
rectangle('Position',[-1/2 -1/2 1 1],'Curvature',[1,1],'EdgeColor',[.6 .6 .6],'LineStyle','-');
rectangle('Position',[-1/4 -1/4 1/2 1/2],'Curvature',[1,1],'EdgeColor',[.6 .6 .6],'LineStyle','-');
plot([0 0],[-5/4 5/4],'Color',[.6 .6 .6],'LineStyle','-');
plot([-5/4 5/4],[0 0],'Color',[.6 .6 .6],'LineStyle','-');
scatter(x(display),y(display),50,color(display,:),'filled')
text(x(display),y(display),labels(display),'HorizontalAlignment','center', 'FontSize',4)
set(gca,'YDir','reverse')
hold off
axis off
daspect([1 1 1])

x0 = 0; y0 = 0; width = 6; height = 6;
set(gcf,'units','centimeters','position',[x0,y0,width,height]);
set(gca, 'FontName', 'Arial');
% set(gca,'FontSize',1);
filepath = '../../paper/Figure 1/';
filename = append(filepath,'electrodes');
print(filename, '-dpng', '-r600');
print(filename, '-depsc', '-r600');
% end