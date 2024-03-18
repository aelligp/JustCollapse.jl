## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------
using JustRelax
import JustRelax.@cell

function copyinn_x!(A, B)

    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    @parallel f_x(A, B)
end
