<a id="html-fe1f6f741c0dc2fbc810"></a>
# The Transistor: History, Physics and Materials

*A short technical overview prepared as a sample document for document parsing.*

<a id="html-6db0e8ee4c37ff31e0ac"></a>
## 1. History

The first working transistor, a **point-contact transistor**, was demonstrated in December 1947 at [Bell Labs](https://en.wikipedia.org/wiki/Bell_Labs) by **John Bardeen** and **Walter Brattain**. Their colleague **William Shockley** developed the theory of the more robust *bipolar junction transistor* in 1948. The three shared the Nobel Prize in Physics in 1956.<sup>1</sup>

The transistor replaced the vacuum tube as the basic switching and amplifying element of electronics, making computers smaller, cheaper and far more reliable.

<a id="html-6a5662424359a3fe4fdb"></a>
### 1.1 Key milestones

| Year | Event | Organisation | People |
| --- | --- | --- | --- |
| 1947 | Point-contact transistor demonstrated | Bell Labs | J. Bardeen, W. Brattain |
| 1948 | Theory of the junction transistor | Bell Labs | W. Shockley |
| 1954 | Commercial silicon transistors | Texas Instruments | G. Teal |
| 1958 | First integrated circuit | Texas Instruments | J. Kilby |
| 1959 | MOSFET invented | Bell Labs | M. Atalla, D. Kahng |
| 1959 | Planar silicon integrated circuit | Fairchild Semiconductor | R. Noyce |
| 1965 | Moore's law formulated | Fairchild Semiconductor | G. Moore |

Table 1. Milestones in transistor technology

<a id="html-de043ef78b713bcf8aae"></a>
## 2. Device physics

<a id="html-b35763de6c084b4078ce"></a>
### 2.1 The p–n junction

The current through an ideal p–n junction diode is described by the **Shockley diode equation**:

$$
I={I}_{S}({e}^{\frac{{V}_{D}}{n{V}_{T}}}-1)
$$

where:

- *I*<sub>S</sub> — reverse saturation current;
- *V*<sub>D</sub> — voltage across the diode;
- *n* — ideality factor, typically between 1 and 2;
- *V*<sub>T</sub> — thermal voltage, given by ${V}_{T}=\frac{kT}{q}\approx 25.85\text{mV}$ at *T* = 300 K.

<a id="html-ed54e6755f373033fb6e"></a>
### 2.2 The MOSFET

In saturation (*V*<sub>DS</sub> ≥ *V*<sub>GS</sub> − *V*<sub>th</sub>), the drain current of a long-channel n-type MOSFET follows the square law:

$$
{I}_{D}=\frac{1}{2}{\mu }_{n}{C}_{ox}\frac{W}{L}{({V}_{GS}-{V}_{th})}^{2}
$$

Shrinking the channel length *L* increases the current and switching speed, which is the physical basis of **Moore's law**. In its simplest form, the number of transistors on a chip doubles every period *T*:

$$
N(t)={N}_{0}⋅{2}^{t/T}
$$

<a id="html-9e4a8fe50ced04698578"></a>
## 3. Semiconductor materials

Silicon dominates digital electronics, while compound semiconductors are used where higher electron mobility or a wider band gap is required.<sup>2</sup>

<table>  <thead> <tr> <th rowspan="2">Group</th> <th rowspan="2">Material</th> <th colspan="3">Electrical properties</th> <th rowspan="2">Typical applications</th> </tr> <tr> <th>Band gap, eV</th> <th>Electron mobility, cm²/(V·s)</th> <th>Breakdown field, MV/cm</th> </tr> </thead> <tbody> <tr><td rowspan="2">Elemental</td><td>Si</td><td>1.12</td><td>1400</td><td>0.3</td><td>CPUs, memory, solar cells</td></tr> <tr><td>Ge</td><td>0.66</td><td>3900</td><td>0.1</td><td>Early transistors, photodetectors</td></tr> <tr><td rowspan="3">Compound</td><td>GaAs</td><td>1.42</td><td>8500</td><td>0.4</td><td>RF amplifiers, LEDs</td></tr> <tr><td>GaN</td><td>3.4</td><td>1000</td><td>3.3</td><td>Power electronics, blue LEDs</td></tr> <tr><td>4H-SiC</td><td>3.26</td><td>900</td><td>2.8</td><td>EV inverters, high-voltage switches</td></tr> </tbody> </table>

Table 2. Properties of common semiconductors at 300 K (approximate values)

eV
0
1
2
3
Si
Ge
GaAs
GaN
4H-SiC

Figure 1. Band gap of common semiconductors at 300 K. Blue: elemental, orange: compound.

<a id="html-a07b5096648033ecaa68"></a>
## 4. Computing a thermal voltage

The thermal voltage from Section 2.1 can be computed directly from physical constants:

```python
k = 1.380649e-23    # Boltzmann constant, J/K
q = 1.602176634e-19 # elementary charge, C
T = 300.0           # temperature, K

v_t = k * T / q
print(f"V_T = {v_t * 1e3:.2f} mV")  # V_T = 25.85 mV
```

<a id="html-c66e5d8468c0876e3497"></a>
## 5. Summary

1. The transistor was invented at Bell Labs in 1947–1948.
2. Two main device families exist:
    - bipolar junction transistors (BJT), controlled by base current;
    - field-effect transistors (FET), controlled by gate voltage:
        - JFET;
        - MOSFET — the basis of modern CMOS logic.
3. Material choice trades off mobility, band gap and cost.

<a id="html-f190afd8339a7788ab18"></a>
### Glossary

Band gap

Energy difference between the top of the valence band and the bottom of the conduction band.

Mobility

How quickly a charge carrier moves through a material under an electric field.

CMOS

Complementary metal–oxide–semiconductor logic built from paired n-type and p-type MOSFETs.

1. The 1956 Nobel Prize in Physics was awarded “for their researches on semiconductors and their discovery of the transistor effect”. ↩
2. Mobility and breakdown values depend on doping and crystal quality; the table lists typical figures. ↩