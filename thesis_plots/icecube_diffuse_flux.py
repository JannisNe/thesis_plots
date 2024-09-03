import numpy as np


def get_joint15():
    contour_68 = np.array(
        [
            (2.362728669872589, 6.014203559565713),
            (2.362693966585692, 5.8056368053145615),
            (2.363985424619503, 5.567299588518169),
            (2.365274403847107, 5.314064746418125),
            (2.371907689256854, 5.180110058995587),
            (2.3798720935997224, 5.04618015963512),
            (2.3905036934212482, 4.942095087006098),
            (2.401135293242774, 4.838010014377076),
            (2.4144390461553713, 4.793565019086808),
            (2.4277551930990033, 4.823608150314808),
            (2.5569555302166473, 5.3176342273561055),
            (2.5742774279906797, 5.422239849288582),
            (2.587615884190174, 5.586361608249467),
            (2.5996232214565462, 5.750458579148281),
            (2.6143051906201973, 5.989093252689504),
            (2.6263274007238113, 6.242575975410243),
            (2.635699766992216, 6.57049724852511),
            (2.637080462049477, 6.8684745426602545),
            (2.6331342025680433, 7.151455059243469),
            (2.6265182688017448, 7.389693123791582),
            (2.617245054781617, 7.657676862822864),
            (2.6066308066035395, 7.866045312577462),
            (2.586711119924644, 8.148728372415844),
            (2.565442962669178, 8.32710326706658),
            (2.5508105696296663, 8.386421099598433),
            (2.528191462991423, 8.44559020375787),
            (2.501561647910366, 8.400401566605522),
            (2.477581676664518, 8.280774379059045),
            (2.456254028060086, 8.10160626642209),
            (2.441577016508849, 7.892766843488175),
            (2.4242476823161963, 7.743468345644738),
            (2.4148951464974466, 7.5347280749591),
            (2.3895245649695105, 7.0575330920628625),
            (2.3788186009617767, 6.714689405582271),
            (2.370787268851321, 6.4463834217440885),
            (2.3680928065043876, 6.2526647166724505),
        ]
    )

    units = 10**-18 / 3.0  # / u.GeV /u.cm**2 / u.s / u.sr
    contour_68.T[1] *= units

    # IceCube Joint Best Fit
    # (https://arxiv.org/abs/1507.03991)
    # all-flabour to muon only
    best_fit_flux = 6.7 * units  # (u.GeV**-1 * u.cm**-2 * u.s**-1 * u.sr**-1)
    best_fit_gamma = 2.5

    # Fit is valid from 25 TeV to 2.8 PeV
    e_range = np.logspace(np.log10(25) + 3, np.log10(2.8) + 6, 100)

    return best_fit_flux, best_fit_gamma, contour_68, e_range


def flux_f(energy, norm, index):
    """Flux function

    :param energy: Energy to evaluate
    :param norm: Flux normalisation at 100 TeV
    :param index: Spectral index
    :return: Flux at given energy
    """
    return norm * (energy**-index) * (10.0**5) ** index


def upper_contour(energy_range, contour):
    """Trace upper contour"""
    return [
        max([flux_f(energy, norm, index) for (index, norm) in contour])
        for energy in energy_range
    ]


def lower_contour(energy_range, contour):
    """Trace lower contour"""
    return [
        min([flux_f(energy, norm, index) for (index, norm) in contour])
        for energy in energy_range
    ]


def get_diffuse_flux_functions(name):

    fluxes = {
        "joint_15": get_joint15,
    }

    best_fit_flux, best_fit_gamma, contour, e_range = fluxes[name]()

    upper_f = lambda e: upper_contour(e, contour)
    lower_f = lambda e: lower_contour(e, contour)
    best_f = lambda e: flux_f(e, best_fit_flux, best_fit_gamma)

    return best_f, lower_f, upper_f, e_range

