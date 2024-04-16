classdef ProximalOperators
    methods
        function self = ProximalOperators()
            % Constructor
        end

        function result = proxL1Norm(self, signal, gamma)
            signalSize = size(signal);
            flatResult = wthresh(signal(:), "s", gamma);
            result = reshape(flatResult, signalSize);
        end

        function result = proxNuclearNorm(self, signal, gamma)
            [u, s, vt] = svd(signal);
            soft_s = zeros(size(signal));
            soft_s(1:min(size(signal)), 1:min(size(signal))) = diag(self.proxL1Norm(diag(s), gamma));

            result = u * soft_s * vt';
        end

        function result = proxBoxConstraint(self, signal, l, r)
            result = min(max(signal, l), r);
        end

    end
end