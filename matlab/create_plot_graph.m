function [G, p] = create_plot_graph(Theta)
    
    dimension = size(Theta,1);
    array = 0:dimension-1;
    alpha = array/dimension*2*pi;
    
    G = graph(Theta, 'OmitSelfLoops');
    LWidths = 4*G.Edges.Weight/max(G.Edges.Weight);
    p = plot(G);
    p.EdgeLabel = G.Edges.Weight;
    p.LineWidth = LWidths;
    p.XData = cos(alpha);
    p.YData = sin(alpha);
    p.MarkerSize = 10;
    
    %,'EdgeLabel',G1.Edges.Weight,'LineWidth',LWidths1);

end